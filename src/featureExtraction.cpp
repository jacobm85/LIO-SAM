// src/featureExtraction.cpp
#include "utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <limits>

struct smoothness_t{
    float value;
    size_t ind;
};

struct by_value{
    bool operator()(smoothness_t const &left, smoothness_t const &right) {
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{
public:
    FeatureExtraction(const rclcpp::NodeOptions & options) :
        ParamServer("lio_sam_featureExtraction", options)
    {
        subLaserCloudInfo = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos,
            std::bind(&FeatureExtraction::laserCloudInfoHandler, this, std::placeholders::_1));

        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos);
        pubCornerPoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/feature/cloud_surface", 1);

        initializationValue();
    }

    ~FeatureExtraction() = default;

private:
    // ROS interfaces
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subLaserCloudInfo;
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPoints;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints;

    // point clouds
    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    // incoming cloud info
    lio_sam::msg::CloudInfo cloudInfo;
    std_msgs::msg::Header cloudHeader;

    // smoothness and labels
    std::vector<smoothness_t> cloudSmoothness;
    std::vector<float> cloudCurvature;
    std::vector<int> cloudNeighborPicked;
    std::vector<int> cloudLabel;

    void initializationValue()
    {
        const size_t max_points = static_cast<size_t>(N_SCAN) * static_cast<size_t>(Horizon_SCAN);

        cloudSmoothness.clear();
        cloudSmoothness.resize(max_points);

        cloudCurvature.clear();
        cloudCurvature.resize(max_points, 0.0f);

        cloudNeighborPicked.clear();
        cloudNeighborPicked.resize(max_points, 0);

        cloudLabel.clear();
        cloudLabel.resize(max_points, 0);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());
    }

    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        cloudInfo = *msgIn; // copy new cloud info
        cloudHeader = msgIn->header; // copy header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // convert deskewed cloud

        // ensure internal buffers are large enough for incoming cloud (defensive)
        const size_t incoming_size = extractedCloud->points.size();
        if (cloudCurvature.size() < incoming_size) {
            cloudCurvature.resize(incoming_size, 0.0f);
            cloudNeighborPicked.resize(incoming_size, 0);
            cloudLabel.resize(incoming_size, 0);
            cloudSmoothness.resize(incoming_size);
        }

        calculateSmoothness();
        markOccludedPoints();
        extractFeatures();
        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        const int cloudSize = static_cast<int>(extractedCloud->points.size());
        if (cloudSize <= 10) return;

        // Initialize first/last 5 elements conservatively
        for (int i = 0; i < std::min(5, cloudSize); ++i) {
            cloudCurvature[i] = std::numeric_limits<float>::max();
            cloudNeighborPicked[i] = 1;
            cloudLabel[i] = 0;
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = static_cast<size_t>(i);
        }
        for (int i = std::max(0, cloudSize - 5); i < cloudSize; ++i) {
            cloudCurvature[i] = std::numeric_limits<float>::max();
            cloudNeighborPicked[i] = 1;
            cloudLabel[i] = 0;
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = static_cast<size_t>(i);
        }

        // Compute curvature-based smoothness for internal points
        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange =
                  cloudInfo.point_range[i-5] + cloudInfo.point_range[i-4]
                + cloudInfo.point_range[i-3] + cloudInfo.point_range[i-2]
                + cloudInfo.point_range[i-1] - cloudInfo.point_range[i] * 10
                + cloudInfo.point_range[i+1] + cloudInfo.point_range[i+2]
                + cloudInfo.point_range[i+3] + cloudInfo.point_range[i+4]
                + cloudInfo.point_range[i+5];

            cloudCurvature[i] = diffRange * diffRange;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = static_cast<size_t>(i);
        }
    }

    void markOccludedPoints()
    {
        const int cloudSize = static_cast<int>(extractedCloud->points.size());
        if (cloudSize <= 10) return;

        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // bounds safety for ranges/columns
            float depth1 = cloudInfo.point_range[i];
            float depth2 = cloudInfo.point_range[i+1];

            int col_i = cloudInfo.point_col_ind[i];
            int col_ip1 = cloudInfo.point_col_ind[i+1];
            int columnDiff = std::abs(col_ip1 - col_i);

            if (columnDiff < 10) {
                if (depth1 - depth2 > 0.3f) {
                    for (int k = -5; k <= 0; ++k) {
                        int idx = i + k;
                        if (idx >= 0 && idx < cloudSize) cloudNeighborPicked[idx] = 1;
                    }
                } else if (depth2 - depth1 > 0.3f) {
                    for (int k = 1; k <= 6; ++k) {
                        int idx = i + k;
                        if (idx >= 0 && idx < cloudSize) cloudNeighborPicked[idx] = 1;
                    }
                }
            }

            // parallel beam check
            float diff1 = std::abs(cloudInfo.point_range[i-1] - cloudInfo.point_range[i]);
            float diff2 = std::abs(cloudInfo.point_range[i+1] - cloudInfo.point_range[i]);

            if (diff1 > 0.02f * cloudInfo.point_range[i] && diff2 > 0.02f * cloudInfo.point_range[i]) {
                cloudNeighborPicked[i] = 1;
            }
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        const int cloudSize = static_cast<int>(extractedCloud->points.size());

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            // clamp start/end indices safely
            int startIdx = 0;
            int endIdx = -1;
            if (i < static_cast<int>(cloudInfo.start_ring_index.size()))
                startIdx = cloudInfo.start_ring_index[i];
            if (i < static_cast<int>(cloudInfo.end_ring_index.size()))
                endIdx = cloudInfo.end_ring_index[i];

            // ensure proper bounds
            startIdx = std::max(0, std::min(startIdx, cloudSize-1));
            endIdx = std::max(-1, std::min(endIdx, cloudSize-1));

            if (endIdx - startIdx < 6) {
                continue; // not enough points in this ring
            }

            // split each scan ring into 6 regions
            for (int j = 0; j < 6; j++)
            {
                int sp = (startIdx * (6 - j) + endIdx * j) / 6;
                int ep = (startIdx * (5 - j) + endIdx * (j + 1)) / 6 - 1;

                if (sp < 0) sp = 0;
                if (ep >= cloudSize) ep = cloudSize - 1;
                if (sp >= ep) continue;

                // sort by smoothness value in this interval
                // ep is inclusive, so pass ep + 1 as end iterator
                std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep + 1, by_value());

                int largestPickedNum = 0;
                // pick edge features from largest (end) to smallest (start)
                for (int k = ep; k >= sp; k--)
                {
                    const int ind = static_cast<int>(cloudSmoothness[k].ind);
                    if (ind < 0 || ind >= cloudSize) continue;

                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 20) {
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            // already have enough edge features for this region
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        // mark neighbors within +/-5, but break on large column jumps
                        for (int l = 1; l <= 5; l++)
                        {
                            int idx = ind + l;
                            if (idx >= cloudSize) break;
                            int colDiff = std::abs(cloudInfo.point_col_ind[idx] - cloudInfo.point_col_ind[idx - 1]);
                            if (colDiff > 10) break;
                            cloudNeighborPicked[idx] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int idx = ind + l;
                            if (idx < 0) break;
                            int colDiff = std::abs(cloudInfo.point_col_ind[idx] - cloudInfo.point_col_ind[idx + 1]);
                            if (colDiff > 10) break;
                            cloudNeighborPicked[idx] = 1;
                        }
                    }
                }

                // pick surface candidates (low curvature)
                for (int k = sp; k <= ep; k++)
                {
                    const int ind = static_cast<int>(cloudSmoothness[k].ind);
                    if (ind < 0 || ind >= cloudSize) continue;

                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {
                            int idx = ind + l;
                            if (idx >= cloudSize) break;
                            int colDiff = std::abs(cloudInfo.point_col_ind[idx] - cloudInfo.point_col_ind[idx - 1]);
                            if (colDiff > 10) break;
                            cloudNeighborPicked[idx] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int idx = ind + l;
                            if (idx < 0) break;
                            int colDiff = std::abs(cloudInfo.point_col_ind[idx] - cloudInfo.point_col_ind[idx + 1]);
                            if (colDiff > 10) break;
                            cloudNeighborPicked[idx] = 1;
                        }
                    }
                }

                // collect unlabelled points as surface candidates
                for (int k = sp; k <= ep; k++)
                {
                    const int ind = static_cast<int>(cloudSmoothness[k].ind);
                    if (ind < 0 || ind >= cloudSize) continue;

                    if (cloudLabel[ind] <= 0) {
                        surfaceCloudScan->push_back(extractedCloud->points[ind]);
                    }
                }
            } // end 6 partitions

            // downsample the surface cloud from this scan line and add
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        } // end N_SCAN loop
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.start_ring_index.clear();
        cloudInfo.end_ring_index.clear();
        cloudInfo.point_col_ind.clear();
        cloudInfo.point_range.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory (to reduce bandwidth when publishing)
        freeCloudInfoMemory();

        // publish extracted corner and surface clouds
        cloudInfo.cloud_corner = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);

        // publish updated cloudInfo (contains pose/flags and now cloud pointers)
        pubLaserCloudInfo->publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto FE = std::make_shared<FeatureExtraction>(options);

    exec.add_node(FE);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Feature Extraction Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
