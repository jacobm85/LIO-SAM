# Cloud Info
std_msgs/Header header 

# Index of the first and last point in each scan ring
int32[] start_ring_index
int32[] end_ring_index

# Point properties
int32[]  point_col_ind   # point column index in range image
float32[] point_range    # point range 

# Sensor availability flags
bool imu_available
bool odom_available

# Attitude for LOAM initialization
float32 imu_roll_init
float32 imu_pitch_init
float32 imu_yaw_init

# Initial guess from IMU pre-integration
float32 initial_guess_x
float32 initial_guess_y
float32 initial_guess_z
float32 initial_guess_roll
float32 initial_guess_pitch
float32 initial_guess_yaw

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original deskewed cloud
sensor_msgs/PointCloud2 cloud_corner    # extracted corner features
sensor_msgs/PointCloud2 cloud_surface   # extracted surface features
