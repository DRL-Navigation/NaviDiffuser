#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose2D.h>
#include "gs_common.h"
#include <gazebo_msgs/GetModelState.h>
#include <comn_pkg/RobotState.h>
#include <comn_pkg/CheckGoal.h>
#include <nav_msgs/Odometry.h>
#include <time.h>
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::LaserScan, nav_msgs::Odometry> MySyncPolicy;

class LocalMap
{
public:
    LocalMap(){}
    void init(int size_x, int size_y, float delta,int width_deviate,int height_deviate)
    {
        delta_ = delta;
        data_ = new unsigned char[size_x * size_y];
        size_x_ = size_x;
        size_y_ = size_y;
        tf_map_base_.setOrigin(tf::Vector3((size_x_ * delta_+2*width_deviate*delta_) / 2.0, (size_y_ * delta_+2*height_deviate* delta_) / 2.0, 0));
        tf::Quaternion q;
        q.setEuler(3.14159, 0, -1.5708);
        tf_map_base_.setRotation(q);
        tf_base_map_ = tf_map_base_.inverse();
        clear();
    }
    void clear()
    {
        for (int i = 0; i < size_x_ * size_y_; i++)
            data_[i] = 5;
    }
    bool free()
    {
        delete data_;
        data_ = NULL;
    }
    ~LocalMap()
    {
        free();
    }
    bool getIndex(float p_x, float p_y, int& x, int& y)
    {
        tf::Vector3 p_base(p_x, p_y, 0);
        tf::Vector3 p_map = tf_base_map_ * p_base;//ADD
        x = p_map.getX() / delta_;
        y = p_map.getY() / delta_;
        if (check_in(x, y))
        {
            return true;
        }
        else
            return false;
    }
    bool check_in(int x, int y)
    {
        return (x >= 0 && x < size_y_ && y >= 0 && y < size_x_);
    }
    bool check_goal_in_view(int x,int y)
    {
        return (x >= 0 && x < size_y_ && y >= 0 && y < size_x_);
    }
    bool check_goal_1_m(int x,int y)
    {
        return (x >= size_y_/4 && x < 3*size_y_/4 && y >= size_x_/4 && y < 3*size_x_/4);
    }
    bool check_goal_1_2_m(int x,int y)
    {
        return (x >= size_y_/12 && x < size_y_/4 && y >= size_x_/12 && y < 11*size_x_/12)||
        (x >= 3*size_y_/4 && x < 11*size_y_/12 && y >= size_x_/12 && y < 11*size_x_/12)||
        (x >= size_y_/4 && x < 3*size_y_/4 && y >= size_x_/12 && y < size_x_/4)||
        (x >= size_y_/4 && x < 3*size_y_/4 && y >= 3*size_x_/4 && y < 11*size_x_/12);
    }
    int getDataId(int x, int y)
    {
        return (y*size_y_ + x);
    }
    void getDataIndex(int d, int& x, int& y)
    {
        x = d % size_y_;
        y = d / size_y_;
    }
    void mapToBase(int x, int y, float& p_x, float& p_y)
    {
        tf::Vector3 p_map(x * delta_, y * delta_, 0);
        tf::Vector3 p_base = tf_map_base_ * p_map;
        p_x = p_base.getX();
        p_y = p_base.getY();
    }
    bool drawCircle(int x, int y, double r, int object_type);
    bool drawRectangle(int x,int y,double a,double b,int object_type);
    unsigned char* data_;
    float delta_;
    int size_x_;
    int size_y_;
    tf::Transform tf_base_map_;
    tf::Transform tf_map_base_;
};

class ScanImage
{
public:
    ScanImage();
    ~ScanImage();
private:

    ros::NodeHandle nh_;
    ros::Subscriber goal_sub_;
    ros::ServiceServer check_goal_srv;
    ros::ServiceClient client_;
    void scanCallback(const sensor_msgs::LaserScan &msg);
    void odomCallback(const nav_msgs::Odometry &msg);
    void goalCallback(const geometry_msgs::PoseStamped &msg);
    void uwbgoalCallback(const geometry_msgs::Pose2D &msg);
    bool checkGoalSrv(comn_pkg::CheckGoal::Request &req,comn_pkg::CheckGoal::Response &res);
    double getSafety(double dx, double dy);
    tf::TransformListener tfl_;
    std::string min_dist_topic_;

    bool dealScan(const sensor_msgs::LaserScan &scan, comn_pkg::RobotState &state_msg, tf::Transform& tf_goal_robot);

    ros::Publisher image_pub_;
    ros::Publisher state_pub_;
    ros::Publisher map_publisher_;
    ros::Publisher min_dist_publisher_;
    LocalMap local_map_;
    std::string laser_frame_;
    int publishMap(LocalMap& map, sensor_msgs::Image& image);

    message_filters::Subscriber<sensor_msgs::LaserScan> *scanSubscriber_;
    message_filters::Subscriber<nav_msgs::Odometry> *odomSubscriber_;
    message_filters::Synchronizer<MySyncPolicy> *syn_;
    ros::Subscriber uwbsub;
    void laser_odom_callback(const sensor_msgs::LaserScanConstPtr &laser, const nav_msgs::OdometryConstPtr &odom);

    int image_width_;
    int image_height_;
    double resolution_;
    double max_dis_;
    int width_deviate;
    int height_deviate;
    std::string laser_topic_;
    std::string goal_topic_;
    std::string check_goal_service_;
    std::string truth_topic_;
    std::string image_topic_;
    std::string odom_topic_;
    std::string state_topic_;
    std::string map_topic_;
    std::string base_frame_;
    std::string track_frame_;
    std::string footprint_frame_;
    std::string robot_name_;
    std::string uwb_goal_topic;
    double angle_min_;
    double angle_max_;
    double min_dist_angle_min_;
    double min_dist_angle_max_;
    double total_time;
    int iter_;
    tf::Stamped<tf::Pose> tf_laser_base_;
    tf::Stamped<tf::Pose> tf_goal_world_;

    bool has_goal_;
    bool draw_goal_;
    bool draw_vw_;
    bool is_rectangle_;
    bool old_weight_;
    std::string env_type_;
    double goal_tolerance_;
    double collision_th_;
    double collision_a_;
    double collision_b_;
    double laser_base_offset_;
    std::string goal_frame_;
    geometry_msgs::Twist cur_vel_;
    geometry_msgs::Pose2D Goal_Postion; //用于UWB实时接受位置
    // multi thread
    int laser_scan_total_;
    sensor_msgs::LaserScan laser_;
    comn_pkg::RobotState state_msg_;
    boost::mutex state_msg_mutex_;
    int thread_total_;
    std::vector<boost::thread*> thread_vector_;
    std::vector<int> thread_status_;
    void deal_scan_thread(int begin_index, int end_index, int thread_num);

    double double_contour_x_;
    double double_contour_y_;
    double rec_angle1;
    double rec_angle2;
    double rec_angle3;
    double rec_angle4;
    double negative_x_shift;
    double negative_y_shift;
    bool bool_subtraction_;
    double deal_subtraction(double cur_angle);
};
