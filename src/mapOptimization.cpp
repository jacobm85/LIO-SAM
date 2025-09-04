// --- ROS2-ported and cleaned up mapOptimization fragment ---
// Includes (kept original + ROS2-safe headers)
#include "utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"
#include "lio_sam/srv/save_map.hpp"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <filesystem>        // safer than system()
#include <cstdlib>           // getenv
#include <sstream>
#include <chrono>
#include <memory>

using namespace gtsam;
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
 */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

// ------------------- mapOptimization class (fragment) -------------------
class mapOptimization : public ParamServer
{
public:
    // GTSAM / pose-graph members
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    // Publishers & subscribers (ROS2)
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryGlobal;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryIncremental;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubHistoryKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegisteredRaw;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopConstraintEdge;

    rclcpp::Service<lio_sam::srv::SaveMap>::SharedPtr srvSaveMap;
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subCloud;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPS;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subLoop;

    // Buffers and containers
    std::deque<nav_msgs::msg::Odometry> gpsQueue;
    lio_sam::msg::CloudInfo cloudInfo;

    std::vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    std::map<int, std::pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

    rclcpp::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    std::map<int, int> loopIndexContainer; // from new to old
    std::vector<std::pair<int, int>> loopIndexQueue;
    std::vector<gtsam::Pose3> loopPoseQueue;
    std::vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    std::deque<std_msgs::msg::Float64MultiArray> loopInfoVec;

    nav_msgs::msg::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    std::unique_ptr<tf2_ros::TransformBroadcaster> br;

    // Constructor: ROS2-safe with filesystem use and proper QoS
    mapOptimization(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_mapOptimization", options)
    {
        // Initialize ISAM2 with reasonable params
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        // Publishers
        pubKeyPoses = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/trajectory", 1);
        pubLaserCloudSurround = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal = create_publisher<nav_msgs::msg::Odometry>("lio_sam/mapping/odometry", qos);
        pubLaserOdometryIncremental = create_publisher<nav_msgs::msg::Odometry>("lio_sam/mapping/odometry_incremental", qos);
        pubPath = create_publisher<nav_msgs::msg::Path>("lio_sam/mapping/path", 1);

        br = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // Subscriptions - use correct QoS (use qos_imu for GPS/IMU-like topics)
        subCloud = create_subscription<lio_sam::msg::CloudInfo>("lio_sam/feature/cloud_info", qos,
            std::bind(&mapOptimization::laserCloudInfoHandler, this, std::placeholders::_1));
        subGPS = create_subscription<nav_msgs::msg::Odometry>(gpsTopic, qos_imu,
            std::bind(&mapOptimization::gpsHandler, this, std::placeholders::_1));
        subLoop = create_subscription<std_msgs::msg::Float64MultiArray>("lio_loop/loop_closure_detection", qos,
            std::bind(&mapOptimization::loopInfoHandler, this, std::placeholders::_1));

        // Service: Save map (ROS2 style). We use a safe callback that uses std::filesystem instead of system()
        srvSaveMap = create_service<lio_sam::srv::SaveMap>(
            "lio_sam/save_map",
            [this](const std::shared_ptr<rmw_request_id_t> /*request_header*/,
                   const std::shared_ptr<lio_sam::srv::SaveMap::Request> req,
                   std::shared_ptr<lio_sam::srv::SaveMap::Response> res)
            {
                // NOTE: rclcpp logging is preferred over std::cout for ROS2
                RCLCPP_INFO(this->get_logger(), "Saving map to pcd files ...");

                // Build save directory: if destination not provided use HOME + savePCDDirectory parameter
                std::string saveMapDirectory;
                const char *home_cstr = std::getenv("HOME");
                std::string home = (home_cstr ? std::string(home_cstr) : std::string("."));

                if (req->destination.empty())
                    saveMapDirectory = home + savePCDDirectory;
                else
                    saveMapDirectory = home + req->destination;

                RCLCPP_INFO(this->get_logger(), "Save destination: %s", saveMapDirectory.c_str());

                try
                {
                    // remove existing dir (if any) and recreate it
                    std::filesystem::path dir(saveMapDirectory);
                    if (std::filesystem::exists(dir))
                        std::filesystem::remove_all(dir);
                    std::filesystem::create_directories(dir);

                    // Save key frame transformations
                    if (cloudKeyPoses3D && !cloudKeyPoses3D->empty())
                    {
                        pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
                    }
                    if (cloudKeyPoses6D && !cloudKeyPoses6D->empty())
                    {
                        pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
                    }

                    // Build global corner/surf clouds by transforming per-keyframe clouds
                    pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
                    pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
                    pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

                    const int nKeys = static_cast<int>(cloudKeyPoses3D ? cloudKeyPoses3D->size() : 0);
                    for (int i = 0; i < nKeys; ++i)
                    {
                        // We assume cornerCloudKeyFrames and surfCloudKeyFrames are maintained elsewhere
                        if (i < static_cast<int>(cornerCloudKeyFrames.size()) && cornerCloudKeyFrames[i])
                        {
                            // transformPointCloud is assumed to exist elsewhere in this class or utility.hpp
                            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
                        }
                        if (i < static_cast<int>(surfCloudKeyFrames.size()) && surfCloudKeyFrames[i])
                        {
                            *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
                        }
                        // Progress log (non-blocking)
                        if ((i % 50) == 0)
                            RCLCPP_INFO(this->get_logger(), "Processing key frame %d / %d", i, nKeys);
                    }

                    // If resolution provided, down-sample before saving
                    if (req->resolution != 0.0)
                    {
                        RCLCPP_INFO(this->get_logger(), "Save resolution: %f", req->resolution);

                        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
                        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());

                        downSizeFilterCorner.setInputCloud(globalCornerCloud);
                        downSizeFilterCorner.setLeafSize(req->resolution, req->resolution, req->resolution);
                        downSizeFilterCorner.filter(*globalCornerCloudDS);
                        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);

                        downSizeFilterSurf.setInputCloud(globalSurfCloud);
                        downSizeFilterSurf.setLeafSize(req->resolution, req->resolution, req->resolution);
                        downSizeFilterSurf.filter(*globalSurfCloudDS);
                        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
                    }
                    else
                    {
                        // save raw combined clouds
                        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
                        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
                    }

                    // Save combined global map
                    *globalMapCloud += *globalCornerCloud;
                    *globalMapCloud += *globalSurfCloud;
                    int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
                    res->success = (ret == 0);

                    // Restore filters leaf sizes to configured mapping params
                    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
                    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

                    RCLCPP_INFO(this->get_logger(), "Saving map to pcd files completed");
                }
                catch (const std::exception &ex)
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to save map: %s", ex.what());
                    res->success = false;
                }
            } // end service lambda
        ); // end create_service

        // More publishers
        pubHistoryKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubLoopConstraintEdge = create_publisher<visualization_msgs::msg::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        // Configure voxel filters using parameters retrieved in ParamServer constructor
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity);

        // Call allocate memory (you should have this function implemented elsewhere)
        allocateMemory();
    } // end constructor

    // ... rest of class (handlers, optimization functions, etc.) ...
}; // end class

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner feature set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf feature set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        // Reserve vectors for maximum expected size to avoid reallocation in hot loops
        const size_t reserveSize = static_cast<size_t>(N_SCAN) * static_cast<size_t>(Horizon_SCAN);
        laserCloudOriCornerVec.resize(reserveSize);
        coeffSelCornerVec.resize(reserveSize);
        laserCloudOriCornerFlag.resize(reserveSize);
        laserCloudOriSurfVec.resize(reserveSize);
        coeffSelSurfVec.resize(reserveSize);
        laserCloudOriSurfFlag.resize(reserveSize);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0.0f;
        }

        matP.setZero();
    }

    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        if (!msgIn) {
            RCLCPP_WARN(get_logger(), "Received null CloudInfo pointer");
            return;
        }

        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = stamp2Sec(msgIn->header.stamp);

        // extract info and feature cloud
        cloudInfo = *msgIn;

        // Safely convert point clouds (check if contained and non-empty)
        if (!msgIn->cloud_corner.data.empty()) {
            try {
                pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
            } catch (const std::exception &e) {
                RCLCPP_ERROR(get_logger(), "Failed to parse cloud_corner: %s", e.what());
                laserCloudCornerLast->clear();
            }
        } else {
            laserCloudCornerLast->clear();
        }

        if (!msgIn->cloud_surface.data.empty()) {
            try {
                pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);
            } catch (const std::exception &e) {
                RCLCPP_ERROR(get_logger(), "Failed to parse cloud_surface: %s", e.what());
                laserCloudSurfLast->clear();
            }
        } else {
            laserCloudSurfLast->clear();
        }

        // main mapping pipeline triggered at configured interval
        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            // Sequence preserved -- algorithmic bodies unchanged
            updateInitialGuess();

            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();

            saveKeyFramesAndFactor();

            correctPoses();

            publishOdometry();

            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::msg::Odometry::SharedPtr gpsMsg)
    {
        if (!gpsMsg) return;
        // small guard: keep gps queue size bounded
        std::lock_guard<std::mutex> lock(mtx);
        gpsQueue.push_back(*gpsMsg);
        while (gpsQueue.size() > 500) gpsQueue.pop_front();
    }

    inline void pointAssociateToMap(const PointType * const pi, PointType * const po) const
    {
        // Use Eigen-style access but keep as plain array multiply for performance
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, const PointTypePose* transformIn)
    {
        auto cloudOut = std::make_shared<pcl::PointCloud<PointType>>();

        if (!cloudIn || cloudIn->empty() || transformIn == nullptr)
            return cloudOut;

        const int cloudSize = static_cast<int>(cloudIn->size());
        cloudOut->resize(cloudSize);

        const Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        // parallelize the point transform - retains your OpenMP usage
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(const PointTypePose &thisPoint) const
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(const float transformIn[]) const
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(const PointTypePose &thisPoint) const
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(const float transformIn[]) const
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(const float transformIn[]) const
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    void visualizeGlobalMapThread()
    {
        // Recommend using a sleep loop rather than spinning a Rate that depends on ROS spinning
        // rclcpp::Rate exists in ROS2, but a simple sleep is clearer in threads
        std::chrono::milliseconds sleepDur(static_cast<long>(1000 / 0.2)); // 0.2 Hz -> 5000 ms
        while (rclcpp::ok())
        {
            std::this_thread::sleep_for(sleepDur);
            // publishGlobalMap should be safe to call periodically (must be implemented elsewhere)
            try {
                publishGlobalMap();
            } catch (const std::exception &e) {
                RCLCPP_WARN(get_logger(), "publishGlobalMap threw exception: %s", e.what());
            }
        }

        // After spinning stops, optionally save PCD if requested
        if (!savePCD)
            return;

        RCLCPP_INFO(get_logger(), "Saving map to pcd files ...");

        // Build save directory safely
        const char *home_cstr = std::getenv("HOME");
        std::string home = (home_cstr ? std::string(home_cstr) : std::string("."));
        std::string outdir = home + savePCDDirectory;

        try
        {
            std::filesystem::path dir(outdir);
            if (std::filesystem::exists(dir))
                std::filesystem::remove_all(dir);
            std::filesystem::create_directories(dir);

            // Save key poses
            if (cloudKeyPoses3D && !cloudKeyPoses3D->empty())
                pcl::io::savePCDFileBinary((outdir + "/trajectory.pcd"), *cloudKeyPoses3D);
            if (cloudKeyPoses6D && !cloudKeyPoses6D->empty())
                pcl::io::savePCDFileBinary((outdir + "/transformations.pcd"), *cloudKeyPoses6D);

            // Build global clouds
            pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

            const int nKeys = static_cast<int>(cloudKeyPoses3D ? cloudKeyPoses3D->size() : 0);
            for (int i = 0; i < nKeys; ++i)
            {
                if (i < static_cast<int>(cornerCloudKeyFrames.size()) && cornerCloudKeyFrames[i] && cloudKeyPoses6D && i < static_cast<int>(cloudKeyPoses6D->size()))
                    *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
                if (i < static_cast<int>(surfCloudKeyFrames.size()) && surfCloudKeyFrames[i] && cloudKeyPoses6D && i < static_cast<int>(cloudKeyPoses6D->size()))
                    *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);

                if ((i % 50) == 0)
                    RCLCPP_INFO(get_logger(), "Processing feature cloud %d of %d", i, nKeys);
            }

            // Downsample and save
            downSizeFilterCorner.setInputCloud(globalCornerCloud);
            downSizeFilterCorner.filter(*globalCornerCloudDS);
            pcl::io::savePCDFileBinary(outdir + "/cloudCorner.pcd", *globalCornerCloudDS);

            downSizeFilterSurf.setInputCloud(globalSurfCloud);
            downSizeFilterSurf.filter(*globalSurfCloudDS);
            pcl::io::savePCDFileBinary(outdir + "/cloudSurf.pcd", *globalSurfCloudDS);

            *globalMapCloud += *globalCornerCloud;
            *globalMapCloud += *globalSurfCloud;
            pcl::io::savePCDFileBinary(outdir + "/cloudGlobal.pcd", *globalMapCloud);

            RCLCPP_INFO(get_logger(), "Saving map to pcd files completed");
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(get_logger(), "Failed saving PCDs: %s", e.what());
        }
    }

    void publishGlobalMap()
    {
        // Safety: do not publish if no subscribers
        if (pubLaserCloudSurround->get_subscription_count() == 0)
            return;

        // Safety: do not publish if no key poses
        if (cloudKeyPoses3D->points.empty())
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;

        // Search near key frames to visualize
        mtx.lock(); // thread safety
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(),
                                      globalMapVisualizationSearchRadius,
                                      pointSearchIndGlobalMap,
                                      pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

        // Downsample near selected key frames (pose density)
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity,
                                                    globalMapVisualizationPoseDensity,
                                                    globalMapVisualizationPoseDensity);
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // Keep intensity (index) consistent after downsampling
        for (auto &pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // Extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
        {
            if (pointDistance(globalMapKeyPosesDS->points[i],
                              cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;

            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                                                        &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],
                                                        &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // Downsample visualized points (leaf size for map visualization)
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize,
                                                     globalMapVisualizationLeafSize,
                                                     globalMapVisualizationLeafSize);
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

        // Publish global map
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }

    void loopClosureThread()
    {
        if (!loopClosureEnableFlag)
            return;

        rclcpp::Rate rate(loopClosureFrequency);
        while (rclcpp::ok())
        {
            rate.sleep();
            performLoopClosure();
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::msg::Float64MultiArray::SharedPtr loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        // Keep buffer small (max 5 entries)
        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        // Copy poses safely for processing
        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // Find candidate loop closure keys
        int loopKeyCur, loopKeyPre;
        if (!detectLoopClosureExternal(&loopKeyCur, &loopKeyPre))
            if (!detectLoopClosureDistance(&loopKeyCur, &loopKeyPre))
                return;

        // Extract clouds for ICP
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);

            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;

            if (pubHistoryKeyFrames->get_subscription_count() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Alignment
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (!icp.hasConverged() || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // Publish corrected cloud
        if (pubIcpKeyFrames->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Compute corrected pose
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame = icp.getFinalTransformation();
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);

        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo   = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);

        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Save loop closure constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        loopIndexContainer[loopKeyCur] = loopKeyPre; // record constraint
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // Skip if already has loop constraint
        if (loopIndexContainer.find(loopKeyCur) != loopIndexContainer.end())
            return false;

        // Find nearest historical keyframe
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(),
                                            historyKeyframeSearchRadius,
                                            pointSearchIndLoop,
                                            pointSearchSqDisLoop, 0);

        for (int id : pointSearchIndLoop)
        {
            if (fabs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;
        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // NOTE: This function is not actively used, but we keep it for compatibility.
        // It provides external loop closure detection based on incoming messages.

        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        // SAFETY: use fabs instead of abs for floating point comparison
        if (fabs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key (find the index corresponding to loopTimeCur)
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = static_cast<int>(round(copy_cloudKeyPoses6D->points[i].intensity));
            else
                break;
        }

        // previous key (find the index corresponding to loopTimePre)
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = static_cast<int>(round(copy_cloudKeyPoses6D->points[i].intensity));
            else
                break;
        }

        // SAFETY: avoid false positive if same key
        if (loopKeyCur == loopKeyPre)
            return false;

        // SAFETY: skip if loop constraint already exists for this key
        if (loopIndexContainer.find(loopKeyCur) != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes,
                               const int& key, const int& searchNum)
    {
        // Extract keyframes around a given index (for ICP loop closure)
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();

        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize)
                continue;

            // Transform both corner and surf keyframes into map frame
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear],
                                                   &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],
                                                   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // Downsample near keyframes for efficiency
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;

        visualization_msgs::msg::MarkerArray markerArray;

        // Marker for loop closure nodes
        visualization_msgs::msg::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::msg::Marker::ADD;
        markerNode.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3;
        markerNode.scale.y = 0.3;
        markerNode.scale.z = 0.3;
        markerNode.color.r = 0;
        markerNode.color.g = 0.8f;
        markerNode.color.b = 1;
        markerNode.color.a = 1;

        // Marker for loop closure edges
        visualization_msgs::msg::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::msg::Marker::ADD;
        markerEdge.type = visualization_msgs::msg::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9f;
        markerEdge.color.g = 0.9f;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        // Build loop closure visualization
        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;

            geometry_msgs::msg::Point p;
            // Current node
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);

            // Previous node
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);

        // Publish loop closure markers
        pubLoopConstraintEdge->publish(markerArray);
    }

    void updateInitialGuess()
    {
        // Save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;

        // Initialization case: no prior keyposes
        if (cloudKeyPoses3D->points.empty())
        {
            transformTobeMapped[0] = cloudInfo.imu_roll_init;
            transformTobeMapped[1] = cloudInfo.imu_pitch_init;
            transformTobeMapped[2] = cloudInfo.imu_yaw_init;

            // SAFETY: allow disabling heading initialization
            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0,
                                                           cloudInfo.imu_roll_init,
                                                           cloudInfo.imu_pitch_init,
                                                           cloudInfo.imu_yaw_init); 
            return;
        }

        // Use IMU pre-integration estimation for pose guess
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odom_available)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(
                cloudInfo.initial_guess_x, cloudInfo.initial_guess_y, cloudInfo.initial_guess_z,
                cloudInfo.initial_guess_roll, cloudInfo.initial_guess_pitch, cloudInfo.initial_guess_yaw);

            if (!lastImuPreTransAvailable)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            }
            else
            {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe  = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;

                pcl::getTranslationAndEulerAngles(transFinal,
                                                  transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                  transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuPreTransformation = transBack;

                // Store IMU transform for incremental update
                lastImuTransformation = pcl::getTransformation(0, 0, 0,
                                                               cloudInfo.imu_roll_init,
                                                               cloudInfo.imu_pitch_init,
                                                               cloudInfo.imu_yaw_init);
                return;
            }
        }

        // Use IMU incremental estimation (rotation only)
        if (cloudInfo.imu_available)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0,
                                                               cloudInfo.imu_roll_init,
                                                               cloudInfo.imu_pitch_init,
                                                               cloudInfo.imu_yaw_init);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;
            Eigen::Affine3f transTobe  = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;

            pcl::getTranslationAndEulerAngles(transFinal,
                                              transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            // Save IMU state for next iteration
            lastImuTransformation = pcl::getTransformation(0, 0, 0,
                                                           cloudInfo.imu_roll_init,
                                                           cloudInfo.imu_pitch_init,
                                                           cloudInfo.imu_yaw_init);
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();

        // Collect latest keyframes up to surroundingKeyframeSize
        for (int i = numPoses - 1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // Extract nearby key poses using kd-tree search
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(),
                                                (double)surroundingKeyframeSearchRadius,
                                                pointSearchInd, pointSearchSqDis);

        for (int id : pointSearchInd)
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);

        // Downsample nearby key poses
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        // Restore intensity (index) after downsampling
        for (auto &pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // Also add some of the latest frames to handle in-place rotation
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses - 1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // Fuse selected keyframes into the local map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();

        for (int i = 0; i < static_cast<int>(cloudToExtract->size()); ++i)
        {
            // SAFETY: skip points too far from the latest pose
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = static_cast<int>(cloudToExtract->points[i].intensity);

            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())
            {
                // Use cached transformed cloud if available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            }
            else
            {
                // Transform and cache the corner/surf keyframe
                pcl::PointCloud<PointType> laserCloudCornerTemp =
                    *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp =
                    *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);

                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;

                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
        }

        // Downsample the fused corner keyframes (map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();

        // Downsample the fused surface keyframes (map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // SAFETY: clear cache if too large to prevent memory blowup
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        // NOTE: loop closure is disabled here, fallback to nearby extraction
        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample current scan (corner + surf)
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        // Update transformation matrix for mapping
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);

            // Find 5 nearest neighbors from map
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            // SAFETY: check distance threshold
            if (pointSearchSqDis[4] < 1.0)
            {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++)
                {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5; cz /= 5;

                // Compute covariance matrix
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++)
                {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5;
                a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                // Check if eigenvalues suggest a line structure
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
                {
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;

                    // Compute two endpoints of line
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // Distance from point to line
                    float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                                      ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                                      ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                                      ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                                      ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                                      ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

                    float l12 = sqrt((x1 - x2) * (x1 - x2) +
                                     (y1 - y2) * (y1 - y2) +
                                     (z1 - z2) * (z1 - z2));

                    float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                                (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;
                    float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
                                 (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;
                    float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                                 (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;
                    float s   = 1 - 0.9f * fabs(ld2); // weight factor

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1f)
                    {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i]      = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);

            // Find 5 nearest neighbors from surf map
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0)
            {
                // Construct plane equation Ax + By + Cz + D = 0
                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid)
                {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    // Weight based on distance to plane and range
                    float s = 1 - 0.9f * fabs(pd2) /
                                   sqrt(sqrt(pointOri.x * pointOri.x +
                                             pointOri.y * pointOri.y +
                                             pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1f)
                    {
                        laserCloudOriSurfVec[i]  = pointOri;
                        coeffSelSurfVec[i]       = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs()
    {
        // Combine edge feature coefficients
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
        {
            if (laserCloudOriCornerFlag[i])
            {
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }

        // Combine surf feature coefficients
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
        {
            if (laserCloudOriSurfFlag[i])
            {
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }

        // Reset flags for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(),   laserCloudOriSurfFlag.end(),   false);
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is adapted from the original LOAM by Ji Zhang.
        // It solves for small pose increments using a linearized system.
        // NOTE: A special lidar<->camera coordinate mapping is used.

        // Extract sine/cosine of current pose (roll, pitch, yaw)
        float srx = sin(transformTobeMapped[1]); // pitch
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]); // yaw
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]); // roll
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            // SAFETY: not enough constraints to solve
            return false;
        }

        // Ax = b system setup
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt, matAtA, matB, matAtB, matX;
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        matAt.create(6, laserCloudSelNum, CV_32F);
        matAtA.create(6, 6, CV_32F);
        matB.create(laserCloudSelNum, 1, CV_32F);
        matAtB.create(6, 1, CV_32F);
        matX.create(6, 1, CV_32F);

        PointType pointOri, coeff;

        // Construct linear system
        for (int i = 0; i < laserCloudSelNum; i++) {
            // Transform into camera frame (custom mapping)
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;

            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;

            // Jacobian components wrt roll, pitch, yaw
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x
                      + (-cry*crz - srx*sry*srz)*pointOri.y) * coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry - cry*srx*srz)*pointOri.y) * coeff.z;

            // Fill A matrix (Jacobian) and B vector (residuals)
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        // Solve normal equations
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // Handle degeneracy in first iteration
        if (iterCount == 0) {
            cv::Mat matE, matV;
            cv::eigen(matAtA, matE, matV);

            cv::Mat matV2 = matV.clone();
            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100}; // thresholds

            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2; // projection matrix for degenerate directions
        }

        // Apply degeneracy correction
        if (isDegenerate) {
            cv::Mat matX2 = matX.clone();
            matX = matP * matX2;
        }

        // Update estimated pose
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        // Convergence check
        float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        // SAFETY: require minimum feature count
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum &&
            laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // Build KD-trees for current local map
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            // Iterative optimization loop
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();
                combineOptimizationCoeffs();

                if (LMOptimization(iterCount)) // converged
                    break;
            }

            transformUpdate();
        }
        else {
            RCLCPP_WARN(get_logger(),
                        "Not enough features! Only %d edge and %d planar features available.",
                        laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        // Fuse IMU into roll/pitch when available
        if (cloudInfo.imu_available)
        {
            if (std::abs(cloudInfo.imu_pitch_init) < 1.4) // exclude upside-down IMU
            {
                double imuWeight = imuRPYWeight;
                tf2::Quaternion imuQuaternion, transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // Slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
                    .getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // Slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
                    .getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        // Apply hard constraints (safeguard against drift)
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        // Update incremental odometry (back side)
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        // Clamp rotation/translation within bounds
        if (value < -limit)
            return -limit;
        if (value > limit)
            return limit;
        return value;
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        // SAFETY: special case for Livox (avoid too frequent keyframes)
        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }

        // Compute relative transform between last and current pose
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(
            transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;

        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // Only save frame if movement exceeds thresholds
        if (std::abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            std::abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            std::abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            // First pose: add a prior to fix the graph reference frame
            noiseModel::Diagonal::shared_ptr priorNoise =
                noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished());
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }
        else
        {
            // Between-factor: connect last keyframe with the new pose
            noiseModel::Diagonal::shared_ptr odometryNoise =
                noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);

            gtSAMgraph.add(BetweenFactor<Pose3>(
                cloudKeyPoses3D->size()-1,
                cloudKeyPoses3D->size(),
                poseFrom.between(poseTo),
                odometryNoise));

            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // Require system initialized
        if (cloudKeyPoses3D->points.empty())
            return;
        // Require some distance travelled before fusing GPS
        if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
            return;

        // If covariance already low, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            double gpsTime = stamp2Sec(gpsQueue.front().header.stamp);
            if (gpsTime < timeLaserInfoCur - 0.2) {
                gpsQueue.pop_front(); // too old
                continue;
            }
            else if (gpsTime > timeLaserInfoCur + 0.2) {
                break; // too new, wait
            }

            nav_msgs::msg::Odometry thisGPS = gpsQueue.front();
            gpsQueue.pop_front();

            // GPS covariance check
            float noise_x = thisGPS.pose.covariance[0];
            float noise_y = thisGPS.pose.covariance[7];
            float noise_z = thisGPS.pose.covariance[14];
            if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                continue;

            // Extract GPS values
            float gps_x = thisGPS.pose.pose.position.x;
            float gps_y = thisGPS.pose.pose.position.y;
            float gps_z = thisGPS.pose.pose.position.z;
            if (!useGpsElevation) {
                gps_z = transformTobeMapped[5]; // lock Z
                noise_z = 0.01;
            }

            // Skip invalid GPS (0,0,0)
            if (fabs(gps_x) < 1e-6 && fabs(gps_y) < 1e-6)
                continue;

            // Only add GPS every ~5m
            PointType curGPSPoint;
            curGPSPoint.x = gps_x;
            curGPSPoint.y = gps_y;
            curGPSPoint.z = gps_z;
            if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                continue;
            lastGPSPoint = curGPSPoint;

            // Build noise model
            gtsam::Vector Vector3(3);
            Vector3 << std::max(noise_x, 1.0f),
                       std::max(noise_y, 1.0f),
                       std::max(noise_z, 1.0f);
            noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);

            // Add GPS factor
            gtsam::GPSFactor gps_factor(
                cloudKeyPoses3D->size(),
                gtsam::Point3(gps_x, gps_y, gps_z),
                gps_noise);
            gtSAMgraph.add(gps_factor);

            aLoopIsClosed = true; // force ISAM2 relinearization
            break;
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (size_t i = 0; i < loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo   = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];

            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        // Reset loop buffers
        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        if (!saveFrame()) // check if movement justifies saving a keyframe
            return;

        // Add new factors
        addOdomFactor();
        addGPSFactor();
        addLoopFactor();

        // Update ISAM2 optimizer
        isam->update(gtSAMgraph, initialEstimate);
        isam->update(); // one extra pass

        if (aLoopIsClosed) {
            // Run multiple extra updates to ensure loop closure propagates
            for (int i = 0; i < 5; i++) {
                isam->update();
            }
        }

        // Clear temporary buffers
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        // === Store key pose results ===
        isamCurrentEstimate = isam->calculateEstimate();
        Pose3 latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

        // Store 3D pose (for visualization)
        PointType thisPose3D;
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // index
        cloudKeyPoses3D->push_back(thisPose3D);

        // Store 6D pose (for future odometry)
        PointTypePose thisPose6D;
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // Update covariance
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // Save updated transform (latest global pose)
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // Store downsampled scan of keyframe
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,   *thisSurfKeyFrame);
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // Append to path (for visualization)
        updatePath(thisPose6D);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        // Publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);

        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);

        // Publish registered key frame
        if (pubRecentKeyFrame->get_subscription_count() > 0)
        {
            auto cloudOut = std::make_shared<pcl::PointCloud<PointType>>();
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,   &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }

        // Publish registered high-res raw cloud
        if (pubCloudRegisteredRaw->get_subscription_count() > 0)
        {
            auto cloudOut = std::make_shared<pcl::PointCloud<PointType>>();
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }

        // Publish path
        if (pubPath->get_subscription_count() > 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath->publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto MO = std::make_shared<mapOptimization>(options);
    exec.add_node(MO);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Map Optimization Started.\033[0m");

    std::thread loopthread(&mapOptimization::loopClosureThread, MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, MO);

    exec.spin();

    rclcpp::shutdown();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
