#include "scan_image.h"
#include <cmath>

ScanImage::ScanImage() :
    scanSubscriber_(NULL),odomSubscriber_(NULL), syn_(NULL)
{
    ros::NodeHandle private_nh("~");
    private_nh.param("env_type", env_type_, std::string("gazebo")); //环境类型： gazebo， stage或者实体real
    private_nh.param("robot_name", robot_name_, std::string("turtlerobot"));
    private_nh.param("image_width", image_width_, 60); //state图像的宽度，格子数
    private_nh.param("image_height", image_height_, 60); //state图像的长度，格子数
    private_nh.param("resolution", resolution_, 0.1); //state图像格子的大小，单位米
    private_nh.param("laser_topic", laser_topic_, std::string("scan")); // 激光topic名称
    private_nh.param("odom_topic", odom_topic_, std::string("odom")); // 里程计topic名称
    private_nh.param("state_topic", state_topic_, std::string("state")); // 发布state的topic名称
    private_nh.param("truth_topic", truth_topic_, std::string("base_pose_ground_truth"));
    private_nh.param("goal_topic", goal_topic_, std::string("goal")); // 接收训练发送的目标点
    private_nh.param("check_goal_service_", check_goal_service_,std::string("check_goal_service"));
    private_nh.param("image_topic", image_topic_, std::string("scan_image")); // 可视化state
    private_nh.param("map_topic", map_topic_, std::string("scan_map")); // 可视化state
    private_nh.param("min_dist_topic", min_dist_topic_, std::string("min_dist"));  // 可视化state
    private_nh.param("base_frame", base_frame_, std::string("base_link"));
    private_nh.param("track_frame", track_frame_, std::string("base_link"));
    private_nh.param("footprint_frame", footprint_frame_, std::string("base_footprint"));
    private_nh.param("laser_frame", laser_frame_, std::string("laser_link"));
    private_nh.param("angle_min", angle_min_, -4.0);
    private_nh.param("angle_max", angle_max_, 4.0);
    private_nh.param("min_dist_angle_min", min_dist_angle_min_, -1.75);
    private_nh.param("min_dist_angle_max", min_dist_angle_max_, 1.75);
    private_nh.param("goal_tolerance", goal_tolerance_, 0.3);
    private_nh.param("collision_th", collision_th_, 0.3);
    private_nh.param("collision_a", collision_a_, 0.7);
    private_nh.param("collision_b", collision_b_, 0.7);
    private_nh.param("is_rectangle", is_rectangle_, false);
    private_nh.param("laser_base_offset", laser_base_offset_, 0.17);
    private_nh.param("thread_total", thread_total_, 12);
    private_nh.param("laser_scan_total", laser_scan_total_, 512);
    private_nh.param("width_deviate", width_deviate, 0);
    private_nh.param("height_deviate", height_deviate, 0);
    private_nh.param("old_weight", old_weight_, true);
    private_nh.param("uwb_goal", uwb_goal_topic, std::string("uwb_goal"));
    // multi shape params =-=- qqc;
    private_nh.param("bool_subtraction", bool_subtraction_, false);
    private_nh.param("double_contour_x", double_contour_x_, 0.17);
    private_nh.param("double_contour_y", double_contour_y_, 0.17);
    negative_x_shift = 0;
    negative_y_shift = 0;
    rec_angle1 = atan2(-double_contour_y_ - negative_y_shift,  double_contour_x_ - negative_x_shift);
    rec_angle2 =  atan2(double_contour_y_ - negative_y_shift, double_contour_x_ - negative_x_shift);
    rec_angle3 =  atan2(double_contour_y_ - negative_y_shift, -double_contour_x_ - negative_x_shift);
    rec_angle4 =  atan2(-double_contour_y_ - negative_y_shift, -double_contour_x_ - negative_x_shift);    
    
    

    thread_vector_.resize(thread_total_);
    thread_status_.resize(thread_total_);
    total_time=0.0;
    iter_=0;
        

    int laser_split = std::ceil(float(laser_scan_total_) / thread_total_);
    for (int i = 0; i < thread_total_; i++)
    {
        int begin_index = laser_split * i;
        int end_index = std::min(laser_split * (i+1), laser_scan_total_);
        thread_vector_[i] = new boost::thread(boost::bind(&ScanImage::deal_scan_thread, this,
                                                          begin_index, end_index, i));
        thread_status_[i] = 0;
    }

    std::cout << "init thread: " << thread_total_ << std::endl;
    local_map_.init(image_width_,image_height_,resolution_,width_deviate,height_deviate);
    double width = image_width_ * resolution_;
    double height = image_height_ * resolution_;
    max_dis_ = sqrt((width+width_deviate*resolution_) * (width+width_deviate*resolution_)  / 4.0 + (height+height_deviate* resolution_) * (height +height_deviate*resolution_)/ 4.0);
    has_goal_ = false;
    draw_goal_ = false;
    draw_vw_ = false;
    client_ = nh_.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    check_goal_srv = nh_.advertiseService(check_goal_service_,&ScanImage::checkGoalSrv,this);
    uwbsub=nh_.subscribe(uwb_goal_topic,10,&ScanImage::uwbgoalCallback,this);
    odomSubscriber_ = new message_filters::Subscriber<nav_msgs::Odometry>(nh_, odom_topic_, 1);
    scanSubscriber_ = new message_filters::Subscriber<sensor_msgs::LaserScan>(nh_, laser_topic_, 1);
//    if (env_type_ != "real")
//    {
//        syn_ = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *scanSubscriber_, *odomSubscriber_);
//        syn_->registerCallback(boost::bind(&ScanImage::laser_odom_callback, this, _1, _2));
//    }
//    else
    {
        odomSubscriber_->registerCallback(&ScanImage::odomCallback, this);
        scanSubscriber_->registerCallback(&ScanImage::scanCallback, this);
    }

    goal_sub_ = nh_.subscribe(goal_topic_, 10, &ScanImage::goalCallback, this);
    image_pub_ = nh_.advertise<sensor_msgs::Image>(image_topic_, 1);
    state_pub_ = nh_.advertise<comn_pkg::RobotState>(state_topic_, 1);
    map_publisher_ = nh_.advertise<nav_msgs::OccupancyGrid>(map_topic_, 1);
    min_dist_publisher_ = nh_.advertise<geometry_msgs::PointStamped>(min_dist_topic_, 1);
    tf_laser_base_.setOrigin(tf::Vector3(laser_base_offset_, 0.0, 0.25));
//    tf_laser_base_.setOrigin(tf::Vector3(0, 0, 0));
    tf_laser_base_.setRotation(tf::Quaternion(0.0, 0.0, 0.0, 1.0));

}

bool LocalMap::drawCircle(int x, int y, double r,int object_type)
{
    bool is_cover= false;
    int r_=ceil(r / delta_);
    for(int i =-r_ ;i<=r_ ;i++)
        for(int j= -r_ ;j<=r_;j++)
            if(i*i+j*j<=r_*r_)
            {
                int pix_x = x+i;
                int pix_y = y+j;
                if (check_in(pix_x,pix_y))
                {
                    if(data_[getDataId(pix_x,pix_y)]==1)
                        is_cover=true;
                    else
                        data_[getDataId(pix_x,pix_y)]=object_type;
                }
            }
    return(is_cover);
}

bool LocalMap::drawRectangle(int x, int y, double a, double b, int object_type)
{
    bool is_cover= false;
    int a_=ceil(a / 2.0 / delta_);
    int b_=ceil(b / 2.0 / delta_);
    for(int i =-a_ ;i<=a_ ;i++)
        for(int j= -b_ ;j<=b_;j++)
            {
                int pix_x = x+i;
                int pix_y = y+j;
                if (check_in(pix_x,pix_y))
                {
                    if(data_[getDataId(pix_x,pix_y)]==1)
                        is_cover=true;
                    else
                        data_[getDataId(pix_x,pix_y)]=object_type;
                }
            }
    return(is_cover);
}

double ScanImage::getSafety(double dx, double dy)
{
    double score_x, score_y;
    if (dx < 0)
        score_x = std::max(0.0, 1+dx/0.3);
    else
        score_x = std::max(0.0, 1-dx/3);
    score_y = std::max(0.0, 1-std::fabs(dy));
    return -(score_x + score_y);
}
void ScanImage::uwbgoalCallback(const geometry_msgs::Pose2D &msg){
    Goal_Postion=msg;
}
bool ScanImage::dealScan(const sensor_msgs::LaserScan &scan, comn_pkg::RobotState &state_msg, tf::Transform& tf_goal_robot)
{
    local_map_.clear();
    state_msg.laser.clear();
    laser_frame_ = scan.header.frame_id;
    state_msg.min_dist.header = scan.header;
    state_msg.min_dist.header.frame_id = track_frame_;
    state_msg.min_dist.point.z = 999;
    state_msg.safety = 0;
    IntPoint p0;
    tf::Vector3 xy_p0, xy_p0_base;
    xy_p0.setValue(0, 0, 0);
    xy_p0_base = tf_laser_base_ * xy_p0;//xy_p0_base -> tf_laser_base_
    local_map_.getIndex(xy_p0_base.getX(), xy_p0_base.getY(), p0.x, p0.y);
    for (int i = 0; i < scan.ranges.size(); i++)
    {
        double d;
        if (std::isnan(scan.ranges[i]))
        {
            state_msg.laser.push_back(0);
            continue;
        }
        else if (std::isinf(scan.ranges[i]))
            d = max_dis_;
        else if (scan.ranges[i] < 0.0015)
            d = max_dis_; //???
        else
            d = scan.ranges[i];
        state_msg.laser.push_back(d);
        double theta = scan.angle_min + scan.angle_increment * i;
        if (theta < angle_min_ || theta > angle_max_)
            continue;
        double x = d * cos(theta);
        double y = d * sin(theta);
        tf::Vector3 xy, xy_base;
        xy.setValue(x, y, 0);
        xy_base = tf_laser_base_ * xy;
        double x_base = xy_base.getX();
        double y_base = xy_base.getY();
        double safety = getSafety(x_base, y_base);
        if (safety < state_msg.safety)
            state_msg.safety = safety;
//        double x_base = x;
//        double y_base = y;
        if (theta >= min_dist_angle_min_ && theta <= min_dist_angle_max_)
        {
            double dist = sqrt(x_base * x_base + y_base * y_base);
            if (dist < state_msg.min_dist.point.z)
            {
                state_msg.min_dist.point.x = x_base;
                state_msg.min_dist.point.y = y_base;
                state_msg.min_dist.point.z = dist;
            }
        }
        IntPoint p1;
        local_map_.getIndex(x_base,y_base, p1.x, p1.y);
        IntPoint linePoints[2000];
        GridLineTraversalLine line;
        line.points = linePoints;
        line.num_points = 0;
        GridLineTraversal::gridLine(p0, p1, &line);
        for (int i = 0; i < line.num_points - 1; i++)
        {
            if (local_map_.check_in(line.points[i].x, line.points[i].y))
            {
                local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] = 8;//free
            }
        }
        if (local_map_.check_in(p1.x, p1.y))
        {
            local_map_.data_[local_map_.getDataId(p1.x, p1.y)] = 1;//obstacle
        }
    }
    float last_goal_x, last_goal_y, tmp_goal_x, tmp_goal_y;
    IntPoint p_goal, p_origin;
    local_map_.getIndex(0, 0, p_origin.x, p_origin.y);
    if (has_goal_ && draw_goal_)
    {
        float goal_x = tf_goal_robot.getOrigin().getX();
        float goal_y = tf_goal_robot.getOrigin().getY();
        local_map_.getIndex(goal_x, goal_y, p_goal.x, p_goal.y);
        if (local_map_.check_in(p_goal.x, p_goal.y))
        {
            last_goal_x = goal_x;
            last_goal_y = goal_y;
            state_msg.goal_out = 0;
        }
        else
        {
            state_msg.goal_out = 1; // goal out map
            IntPoint linePoints[2000];
            GridLineTraversalLine line;
            line.points = linePoints;
            line.num_points = 0;
            GridLineTraversal::gridLine(p0, p_goal, &line);
            double max_dist = 0;
            for (int i = 0; i < line.num_points - 1; i++)
            {
                if (local_map_.check_in(line.points[i].x, line.points[i].y))
                {
                    local_map_.mapToBase(line.points[i].x, line.points[i].y, tmp_goal_x, tmp_goal_y);
                    double tmp_dist = (tmp_goal_x*tmp_goal_x+tmp_goal_y*tmp_goal_y);
                    if (tmp_dist > max_dist)
                    {
                        max_dist = tmp_dist;
                        last_goal_x = tmp_goal_x;
                        last_goal_y = tmp_goal_y;
                    }
                }
            }
        }
    }
    min_dist_publisher_.publish(state_msg.min_dist);
    state_msg.laser_image.encoding = "8UC1";
    state_msg.laser_image.height = image_height_;
    state_msg.laser_image.width = image_width_;
    state_msg.laser_image.step = image_height_ * 1;
    state_msg.laser_image.header = scan.header;
    state_msg.laser_image.header.frame_id = track_frame_;
    int image_size = image_height_ * image_width_;
    state_msg.laser_image.data.resize(image_size);
    state_msg.collision = 0;

    bool goal_collision,robot_collision;
    if (has_goal_ && draw_goal_ )
        goal_collision=local_map_.drawCircle(p_goal.x, p_goal.y, goal_tolerance_,10);
    if(is_rectangle_)
        robot_collision=local_map_.drawRectangle(p_origin.x,p_origin.y, collision_a_, collision_b_, 3);  
    else
        robot_collision=local_map_.drawCircle(p_origin.x,p_origin.y, collision_th_, 3);
    state_msg.collision = robot_collision;

    if (draw_vw_)
    {
        float x_v,y_v;
        IntPoint p_v,p_w;
        if(is_rectangle_)
        {
            x_v = state_msg.velocity.linear.x+collision_a_/2.0+resolution_;
            y_v = state_msg.velocity.angular.z*collision_a_;
            local_map_.getIndex(x_v,0, p_v.x, p_v.y);
            local_map_.getIndex(collision_a_/2.0+resolution_,y_v, p_w.x, p_w.y);
            local_map_.getIndex(collision_a_/2.0+resolution_,0, p0.x, p0.y);
        }
        else
        {
            x_v = state_msg.velocity.linear.x+collision_th_;
            y_v = state_msg.velocity.angular.z*collision_th_*2;
            local_map_.getIndex(x_v,0, p_v.x, p_v.y);
            local_map_.getIndex(collision_th_,y_v, p_w.x, p_w.y);
            local_map_.getIndex(collision_th_,0, p0.x, p0.y);
        }
        IntPoint linePoints[1000];
        GridLineTraversalLine line;
        line.points = linePoints;
        line.num_points = 0;
        GridLineTraversal::gridLine(p0, p_v, &line);
        for (int i = 0; i < line.num_points - 1; i++)
        {
            if (local_map_.check_in(line.points[i].x, line.points[i].y))
            {
                local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] -=1; //velocity
            }
        }
        line.num_points = 0;
        GridLineTraversal::gridLine(p0, p_w, &line);
        for (int i = 0; i < line.num_points - 1; i++)
        {
            if (local_map_.check_in(line.points[i].x, line.points[i].y))
            {
                local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] -= 1; //velocity
            }
        }
    }
    //local_map_data
    //0=obs+vel,1=obs
    //2=rob+vel,3=rob
    //4=unknow+vel,5=unknow
    //7=free+vel,8=free
    //9=goal+vel,10=goal
    for (int m = 0; m < image_size; m++)
    {
        state_msg.laser_image.data[m] = 25*local_map_.data_[m];
    }
    for (int m = 0; m < image_size; m++)
    {
        
    switch(state_msg.laser_image.data[m]){
        case 125:state_msg.laser_image.data[m]=200;break; // 未知
        case 25:state_msg.laser_image.data[m]=0;break;  //障碍物
        case 75:state_msg.laser_image.data[m]=100;break;  //机器人
        default : state_msg.laser_image.data[m]=255;     //空白
    }

    }
    image_pub_.publish(state_msg.laser_image);
    publishMap(local_map_ ,state_msg.laser_image);
    //state_msg.laser = scan;
    return true;
}

double ScanImage::deal_subtraction(double cur_angle){
    double subtraction;
    if (is_rectangle_){
        if ( cur_angle >= rec_angle1 && cur_angle <= rec_angle2 ){
                        subtraction = abs( (double_contour_x_ - negative_x_shift) / cos(cur_angle));
    //                    std::cout << " 11 " << cur_angle << " " << abs( (double_contour_x_ - negative_x_shift) / cos(cur_angle)) << std::endl;
                    }
                    else if( cur_angle >= rec_angle3 || cur_angle <= rec_angle4){
                        subtraction = abs( (-double_contour_x_ - negative_x_shift) / cos(cur_angle));
    //                    std::cout << " 33 " << cur_angle << " " << abs( (-double_contour_x_ - negative_x_shift) / cos(cur_angle)) << std::endl;
                    }
                    else if( cur_angle > rec_angle2 && cur_angle < rec_angle3){
                        subtraction = abs( (double_contour_y_ - negative_y_shift) / sin(cur_angle));
    //                    std::cout <<  " 22 " << cur_angle << " " << abs( (double_contour_y_ - negative_y_shift) / sin(cur_angle)) << std::endl;
                    }
                    else{
                        subtraction = abs( (-double_contour_y_ - negative_y_shift) / sin(cur_angle));
    //                    std::cout <<  " 44 " << cur_angle << " " << abs( (-double_contour_y_ - negative_y_shift) / sin(cur_angle)) << std::endl;
                    }
//                    std::cout << cur_angle << " " << subtraction << std::endl;
    }
    else{
        subtraction = double_contour_x_;
    }
    return subtraction;

}

void ScanImage::deal_scan_thread(int begin_index, int end_index, int thread_num)
{
    ros::Rate r(40);
    while(thread_status_[thread_num] != -1)
    {
        if (thread_status_[thread_num] == 1)
        {
            IntPoint p0;
            tf::Vector3 xy_p0, xy_p0_base;
            xy_p0.setValue(0, 0, 0);
            xy_p0_base = tf_laser_base_ * xy_p0;//xy_p0_base -> tf_laser_base_
            local_map_.getIndex(xy_p0_base.getX(), xy_p0_base.getY(), p0.x, p0.y);
            for (int i = begin_index; i < end_index; i++)
            {
                double d;
                if (std::isnan(laser_.ranges[i]))
                {
                    state_msg_.laser[i] = 0;
                    continue;
                }
                else if (std::isinf(laser_.ranges[i]))
                    d = max_dis_;
                else if (laser_.ranges[i] < 0.0015)
                    d = max_dis_; //???
                else
                    d = laser_.ranges[i];
                // state_msg_.laser[i] = d;
                double theta = laser_.angle_min + laser_.angle_increment * i;
                if (theta < angle_min_ || theta > angle_max_)
                    continue;
                double x = d * cos(theta);
                double y = d * sin(theta);
                tf::Vector3 xy, xy_base;
                xy.setValue(x, y, 0);
                xy_base = tf_laser_base_ * xy;
                double x_base = xy_base.getX();
                double y_base = xy_base.getY();

                double subtraction;
                if (bool_subtraction_)
                    subtraction = deal_subtraction(theta);
                else
                    subtraction = 0;
                // qqc multi shape paper.
                state_msg_.laser[i] = sqrt(x_base * x_base + y_base * y_base) - subtraction;

                double safety = getSafety(x_base, y_base);
                if (safety < state_msg_.safety)
                    state_msg_.safety = safety;
        //        double x_base = x;
        //        double y_base = y;
                if (theta >= min_dist_angle_min_ && theta <= min_dist_angle_max_)
                {
                    double dist = sqrt(x_base * x_base + y_base * y_base);
                    if (dist < state_msg_.min_dist.point.z)
                    {
                        state_msg_.min_dist.point.x = x_base;
                        state_msg_.min_dist.point.y = y_base;
                        state_msg_.min_dist.point.z = dist;
                    }
                }
                IntPoint p1;
                local_map_.getIndex(x_base,y_base, p1.x, p1.y);
                IntPoint linePoints[2000];
                GridLineTraversalLine line;
                line.points = linePoints;
                line.num_points = 0;
                GridLineTraversal::gridLine(p0, p1, &line);
                for (int i = 0; i < line.num_points - 1; i++)
                {
                    if (local_map_.check_in(line.points[i].x, line.points[i].y))
                    {
                        local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] = 8;//free
                    }
                }
                if (local_map_.check_in(p1.x, p1.y))
                {
                    local_map_.data_[local_map_.getDataId(p1.x, p1.y)] = 1;//obstacle
                }
            }
            thread_status_[thread_num] = 0;
        }
        r.sleep();
    }
}

void ScanImage::odomCallback(const nav_msgs::Odometry &msg)
{
    cur_vel_ = msg.twist.twist;
}

void ScanImage::scanCallback(const sensor_msgs::LaserScan &msg)
{   
    clock_t start_t, end_t;
    start_t = clock();
    if (int(msg.ranges.size()) != laser_scan_total_)
    {
        std::cout << msg.ranges.size() << std::endl;
        ROS_ERROR("laser_scan_total param is wrong!");
    }
    tf::Transform tf_goal_robot;
    tf::Stamped<tf::Pose> tf_robot_world;
    if (has_goal_ && goal_tolerance_ != 0.0)
    {
        state_msg_.velocity = cur_vel_;
        if (env_type_ == "gazebo")
        {
            gazebo_msgs::GetModelState srv;
            srv.request.model_name = robot_name_;
            if(client_.call(srv) && srv.response.success)
            {
              geometry_msgs::Pose robot_pose = srv.response.pose;
              state_msg_.pose_world = robot_pose;
              tf_robot_world.setOrigin(tf::Vector3(robot_pose.position.x, robot_pose.position.y, 0));
              tf_robot_world.setRotation(tf::Quaternion(robot_pose.orientation.x, robot_pose.orientation.y,
                                                        robot_pose.orientation.z, robot_pose.orientation.w));
              //state_msg_.velocity = srv.response.twist;
              tf_goal_robot = tf_robot_world.inverse() * tf_goal_world_;
            }
            else
                return;
        }
        else if (env_type_ == "real" || env_type_ == "stage")
        {
            tf::Stamped<tf::Pose> ident(tf::Transform(tf::createIdentityQuaternion(),
                                                      tf::Vector3(0, 0, 0)), msg.header.stamp, track_frame_);
            try
            {
              tfl_.waitForTransform(goal_frame_, track_frame_, msg.header.stamp, ros::Duration(1.0));
              tfl_.transformPose(goal_frame_, ident, tf_robot_world);
            }
            catch (tf::TransformException& e)
            {
              ROS_ERROR("Couldn't transform from %s to %s, "
              "even though shit", track_frame_.c_str(), goal_frame_.c_str());
              return;
            }
            tf_goal_robot = tf_robot_world.inverse() * tf_goal_world_;
        }
        else if(env_type_ == "uwb")
        {   
            tf_goal_robot.setOrigin(tf::Vector3(Goal_Postion.x, Goal_Postion.y, 0));
            tf::Quaternion temp;
            temp.setEuler(0,0,Goal_Postion.theta);
            tf_goal_robot.setRotation(temp);
        }
        //tf_goal_robot = tf_robot_world.inverse() * tf_goal_world_;
        state_msg_.pose.position.x = tf_goal_robot.getOrigin().getX();
        state_msg_.pose.position.y = tf_goal_robot.getOrigin().getY();
        state_msg_.pose.position.z = tf_goal_robot.getOrigin().getZ();
        state_msg_.pose.orientation.x = tf_goal_robot.getRotation().getX();
        state_msg_.pose.orientation.y = tf_goal_robot.getRotation().getY();
        state_msg_.pose.orientation.z = tf_goal_robot.getRotation().getZ();
        state_msg_.pose.orientation.w = tf_goal_robot.getRotation().getW();
    }
    local_map_.clear();
    laser_frame_ = msg.header.frame_id;
    state_msg_.laser_raw = msg;
    state_msg_.min_dist.header = msg.header;
    state_msg_.min_dist.header.frame_id = track_frame_;
    state_msg_.min_dist.point.z = 999;
    state_msg_.safety = 0;
    state_msg_.laser.resize(msg.ranges.size());
    IntPoint p0;
    tf::Vector3 xy_p0, xy_p0_base;
    xy_p0.setValue(0, 0, 0);
    xy_p0_base = tf_laser_base_ * xy_p0;//xy_p0_base -> tf_laser_base_
    local_map_.getIndex(xy_p0_base.getX(), xy_p0_base.getY(), p0.x, p0.y);
    laser_ = msg;
    for (int i = 0; i < thread_total_; i++)
    {
        thread_status_[i] = 1;
    }

    while (true) {
        int finish_num = 0;
        for (int i = 0; i < thread_total_; i++)
        {
            if (thread_status_[i] == 0)
            {
                finish_num += 1;
            }
        }
        if (finish_num == thread_total_)
            break;
    }

    float last_goal_x, last_goal_y, tmp_goal_x, tmp_goal_y;
    IntPoint p_goal, p_origin;
    local_map_.getIndex(0, 0, p_origin.x, p_origin.y);
    if (has_goal_ && draw_goal_)
    {
        float goal_x = tf_goal_robot.getOrigin().getX();
        float goal_y = tf_goal_robot.getOrigin().getY();
        local_map_.getIndex(goal_x, goal_y, p_goal.x, p_goal.y);
        if (local_map_.check_in(p_goal.x, p_goal.y))
        {
            last_goal_x = goal_x;
            last_goal_y = goal_y;
            state_msg_.goal_out = 0;
        }
        else
        {
            state_msg_.goal_out = 1; // goal out map
            IntPoint linePoints[2000];
            GridLineTraversalLine line;
            line.points = linePoints;
            line.num_points = 0;
            GridLineTraversal::gridLine(p0, p_goal, &line);
            double max_dist = 0;
            for (int i = 0; i < line.num_points - 1; i++)
            {
                if (local_map_.check_in(line.points[i].x, line.points[i].y))
                {
                    local_map_.mapToBase(line.points[i].x, line.points[i].y, tmp_goal_x, tmp_goal_y);
                    double tmp_dist = (tmp_goal_x*tmp_goal_x+tmp_goal_y*tmp_goal_y);
                    if (tmp_dist > max_dist)
                    {
                        max_dist = tmp_dist;
                        last_goal_x = tmp_goal_x;
                        last_goal_y = tmp_goal_y;
                    }
                }
            }
        }
    }
    min_dist_publisher_.publish(state_msg_.min_dist);
    state_msg_.laser_image.encoding = "8UC1";
    state_msg_.laser_image.height = image_height_;
    state_msg_.laser_image.width = image_width_;
    state_msg_.laser_image.step = image_height_ * 1;
    state_msg_.laser_image.header = msg.header;
    state_msg_.laser_image.header.frame_id = track_frame_;
    int image_size = image_height_ * image_width_;
    state_msg_.laser_image.data.resize(image_size);
    state_msg_.collision = 0;

    if (draw_vw_)
    {
        float x_v,y_v;
        IntPoint p_v,p_w;
        if(is_rectangle_)
        {
            x_v = state_msg_.velocity.linear.x+collision_a_/2.0+resolution_;
            y_v = state_msg_.velocity.angular.z*collision_a_;
            local_map_.getIndex(x_v,0, p_v.x, p_v.y);
            local_map_.getIndex(collision_a_/2.0+resolution_,y_v, p_w.x, p_w.y);
            local_map_.getIndex(collision_a_/2.0+resolution_,0, p0.x, p0.y);
        }
        else
        {
            x_v = state_msg_.velocity.linear.x+collision_th_;
            y_v = state_msg_.velocity.angular.z*collision_th_*2;
            local_map_.getIndex(x_v,0, p_v.x, p_v.y);
            local_map_.getIndex(collision_th_,y_v, p_w.x, p_w.y);
            local_map_.getIndex(collision_th_,0, p0.x, p0.y);
        }
        IntPoint linePoints[1000];
        GridLineTraversalLine line;
        line.points = linePoints;
        line.num_points = 0;
        GridLineTraversal::gridLine(p0, p_v, &line);
        for (int i = 0; i < line.num_points - 1; i++)
        {
            if (local_map_.check_in(line.points[i].x, line.points[i].y))
            {
                local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] -=1; //velocity
            }
        }
        line.num_points = 0;
        GridLineTraversal::gridLine(p0, p_w, &line);
        for (int i = 0; i < line.num_points - 1; i++)
        {
            if (local_map_.check_in(line.points[i].x, line.points[i].y))
            {
                local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] -= 1; //velocity
            }
        }
    }

    bool goal_collision,robot_collision;
    if(is_rectangle_)
        robot_collision=local_map_.drawRectangle(p_origin.x,p_origin.y, collision_a_, collision_b_, 3);
    else
        robot_collision=local_map_.drawCircle(p_origin.x,p_origin.y, collision_th_, 3);
    state_msg_.collision = robot_collision;

    state_msg_.image_no_goal = state_msg_.laser_image;
    for (int m = 0; m < image_size; m++)
    {
        state_msg_.image_no_goal.data[m] = 25*local_map_.data_[m];
    }

    if (has_goal_ && draw_goal_ )
    {
        goal_collision=local_map_.drawCircle(p_goal.x, p_goal.y, goal_tolerance_,10);

        tf::Vector3 angle_clock;
        angle_clock.setX(goal_tolerance_);
        tf::Vector3 angle_clock_base = tf_goal_robot * angle_clock;
        IntPoint p_clock;
        local_map_.getIndex(angle_clock_base.getX(), angle_clock_base.getY(), p_clock.x, p_clock.y);

        IntPoint linePoints[1000];
        GridLineTraversalLine line;
        line.points = linePoints;
        line.num_points = 0;
        GridLineTraversal::gridLine(p_goal, p_clock, &line);
        for (int i = 0; i < line.num_points - 1; i++)
        {
            if (local_map_.check_in(line.points[i].x, line.points[i].y))
            {
                local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] -=1; //angle
            }
        }
        tf::Transform robot_angle = tf_goal_robot;
        robot_angle.setOrigin(tf::Vector3(0,0,0));
        angle_clock.setX(image_height_ / 2.0 * resolution_);
        angle_clock_base = robot_angle * angle_clock;
        local_map_.getIndex(angle_clock_base.getX(), angle_clock_base.getY(), p_clock.x, p_clock.y);
        line.num_points = 0;
        GridLineTraversal::gridLine(p_origin, p_clock, &line);
        for (int i = 0; i < line.num_points - 1; i++)
        {
            if (local_map_.check_in(line.points[i].x, line.points[i].y))
            {
                local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] -= 1; //velocity
            }
        }

    }

    //local_map_data
    //0=obs+vel,1=obs
    //2=rob+vel,3=rob
    //4=unknow+vel,5=unknow
    //7=free+vel,8=free
    //9=goal+vel,10=goal
    if (old_weight_)
        for (int m = 0; m < image_size; m++)
    {
        state_msg_.laser_image.data[m] = 25*local_map_.data_[m];
    }
    else
    for (int m = 0; m < image_size; m++)
    {   
        
        switch(local_map_.data_[m]){
        case 5:state_msg_.laser_image.data[m]=200;break; // 未知
        case 1:state_msg_.laser_image.data[m]=0;break;  //障碍物
        case 3:state_msg_.laser_image.data[m]=100;break;  //机器人
        case 8:state_msg_.laser_image.data[m]=255;break;//空白
        default : state_msg_.laser_image.data[m]=255;     //空白
    }
    }

    publishMap(local_map_ ,state_msg_.laser_image);
    image_pub_.publish(state_msg_.laser_image);
    
    state_pub_.publish(state_msg_);
    end_t = clock();
    if (iter_==0)
    iter_++;
    else
    {   iter_++;
        total_time+=(double)(end_t - start_t) / CLOCKS_PER_SEC;
//        ROS_INFO("total time is %lf",total_time/(iter_-1));
    }    
}

void ScanImage::laser_odom_callback(const sensor_msgs::LaserScanConstPtr &laser,
                                    const nav_msgs::OdometryConstPtr &odom)
{
    if (int(laser->ranges.size()) != laser_scan_total_)
    {
        ROS_ERROR("laser_scan_total param is wrong!");
    }
    tf::Transform tf_goal_robot;
    tf::Stamped<tf::Pose> tf_robot_world;
    if (has_goal_ && goal_tolerance_ != 0.0)
    {
        state_msg_.velocity = odom->twist.twist;
        if (env_type_ == "gazebo")
        {
            gazebo_msgs::GetModelState srv;
            srv.request.model_name = robot_name_;
            if(client_.call(srv) && srv.response.success)
            {
              geometry_msgs::Pose robot_pose = srv.response.pose;
              tf_robot_world.setOrigin(tf::Vector3(robot_pose.position.x, robot_pose.position.y, 0));
              tf_robot_world.setRotation(tf::Quaternion(robot_pose.orientation.x, robot_pose.orientation.y,
                                                        robot_pose.orientation.z, robot_pose.orientation.w));
              //state_msg_.velocity = srv.response.twist;
            }
            else
                return;
        }
        else if (env_type_ == "real" || env_type_ == "stage")
        {
            tf::Stamped<tf::Pose> ident(tf::Transform(tf::createIdentityQuaternion(),
                                                      tf::Vector3(0, 0, 0)), laser->header.stamp, track_frame_);
            try
            {
              tfl_.waitForTransform(goal_frame_, track_frame_, laser->header.stamp, ros::Duration(1.0));
              tfl_.transformPose(goal_frame_, ident, tf_robot_world);
            }
            catch (tf::TransformException& e)
            {
              ROS_ERROR("Couldn't transform from %s to %s, "
              "even though the message notifier is in use", track_frame_.c_str(), goal_frame_.c_str());
              return;
            }
        }
        tf_goal_robot = tf_robot_world.inverse() * tf_goal_world_;
        state_msg_.pose.position.x = tf_goal_robot.getOrigin().getX();
        state_msg_.pose.position.y = tf_goal_robot.getOrigin().getY();
        state_msg_.pose.position.z = tf_goal_robot.getOrigin().getZ();
        state_msg_.pose.orientation.x = tf_goal_robot.getRotation().getX();
        state_msg_.pose.orientation.y = tf_goal_robot.getRotation().getY();
        state_msg_.pose.orientation.z = tf_goal_robot.getRotation().getZ();
        state_msg_.pose.orientation.w = tf_goal_robot.getRotation().getW();
    }
    local_map_.clear();
    laser_frame_ = laser->header.frame_id;
    state_msg_.min_dist.header = laser->header;
    state_msg_.min_dist.header.frame_id = track_frame_;
    state_msg_.min_dist.point.z = 999;
    state_msg_.safety = 0;
    state_msg_.laser.resize(laser->ranges.size());
    IntPoint p0;
    tf::Vector3 xy_p0, xy_p0_base;
    xy_p0.setValue(0, 0, 0);
    xy_p0_base = tf_laser_base_ * xy_p0;//xy_p0_base -> tf_laser_base_
    local_map_.getIndex(xy_p0_base.getX(), xy_p0_base.getY(), p0.x, p0.y);
    laser_ = *laser;
    for (int i = 0; i < thread_total_; i++)
    {
        thread_status_[i] = 1;
    }

    while (true) {
        int finish_num = 0;
        for (int i = 0; i < thread_total_; i++)
        {
            if (thread_status_[i] == 0)
            {
                finish_num += 1;
            }
        }
        if (finish_num == thread_total_)
            break;
    }

    float last_goal_x, last_goal_y, tmp_goal_x, tmp_goal_y;
    IntPoint p_goal, p_origin;
    local_map_.getIndex(0, 0, p_origin.x, p_origin.y);
    if (has_goal_ && draw_goal_)
    {
        float goal_x = tf_goal_robot.getOrigin().getX();
        float goal_y = tf_goal_robot.getOrigin().getY();
        local_map_.getIndex(goal_x, goal_y, p_goal.x, p_goal.y);
        if (local_map_.check_in(p_goal.x, p_goal.y))
        {
            last_goal_x = goal_x;
            last_goal_y = goal_y;
            state_msg_.goal_out = 0;
        }
        else
        {
            state_msg_.goal_out = 1; // goal out map
            IntPoint linePoints[2000];
            GridLineTraversalLine line;
            line.points = linePoints;
            line.num_points = 0;
            GridLineTraversal::gridLine(p0, p_goal, &line);
            double max_dist = 0;
            for (int i = 0; i < line.num_points - 1; i++)
            {
                if (local_map_.check_in(line.points[i].x, line.points[i].y))
                {
                    local_map_.mapToBase(line.points[i].x, line.points[i].y, tmp_goal_x, tmp_goal_y);
                    double tmp_dist = (tmp_goal_x*tmp_goal_x+tmp_goal_y*tmp_goal_y);
                    if (tmp_dist > max_dist)
                    {
                        max_dist = tmp_dist;
                        last_goal_x = tmp_goal_x;
                        last_goal_y = tmp_goal_y;
                    }
                }
            }
        }
    }
    min_dist_publisher_.publish(state_msg_.min_dist);
    state_msg_.laser_image.encoding = "8UC1";
    state_msg_.laser_image.height = image_height_;
    state_msg_.laser_image.width = image_width_;
    state_msg_.laser_image.step = image_height_ * 1;
    state_msg_.laser_image.header = laser->header;
    state_msg_.laser_image.header.frame_id = track_frame_;
    int image_size = image_height_ * image_width_;
    state_msg_.laser_image.data.resize(image_size);
    state_msg_.collision = 0;

    bool goal_collision,robot_collision;
    if (has_goal_ && draw_goal_ )
        goal_collision=local_map_.drawCircle(p_goal.x, p_goal.y, goal_tolerance_,10);
    if(is_rectangle_)
        robot_collision=local_map_.drawRectangle(p_origin.x,p_origin.y, collision_a_, collision_b_, 3);
    else
        robot_collision=local_map_.drawCircle(p_origin.x,p_origin.y, collision_th_, 3);
    state_msg_.collision = robot_collision;

    if (draw_vw_)
    {
        float x_v,y_v;
        IntPoint p_v,p_w;
        if(is_rectangle_)
        {
            x_v = state_msg_.velocity.linear.x+collision_a_/2.0+resolution_;
            y_v = state_msg_.velocity.angular.z*collision_a_;
            local_map_.getIndex(x_v,0, p_v.x, p_v.y);
            local_map_.getIndex(collision_a_/2.0+resolution_,y_v, p_w.x, p_w.y);
            local_map_.getIndex(collision_a_/2.0+resolution_,0, p0.x, p0.y);
        }
        else
        {
            x_v = state_msg_.velocity.linear.x+collision_th_;
            y_v = state_msg_.velocity.angular.z*collision_th_*2;
            local_map_.getIndex(x_v,0, p_v.x, p_v.y);
            local_map_.getIndex(collision_th_,y_v, p_w.x, p_w.y);
            local_map_.getIndex(collision_th_,0, p0.x, p0.y);
        }
        IntPoint linePoints[1000];
        GridLineTraversalLine line;
        line.points = linePoints;
        line.num_points = 0;
        GridLineTraversal::gridLine(p0, p_v, &line);
        for (int i = 0; i < line.num_points - 1; i++)
        {
            if (local_map_.check_in(line.points[i].x, line.points[i].y))
            {
                local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] -=1; //velocity
            }
        }
        line.num_points = 0;
        GridLineTraversal::gridLine(p0, p_w, &line);
        for (int i = 0; i < line.num_points - 1; i++)
        {
            if (local_map_.check_in(line.points[i].x, line.points[i].y))
            {
                local_map_.data_[local_map_.getDataId(line.points[i].x, line.points[i].y)] -= 1; //velocity
            }
        }
    }
    //local_map_data
    //0=obs+vel,1=obs
    //2=rob+vel,3=rob
    //4=unknow+vel,5=unknow
    //7=free+vel,8=free
    //9=goal+vel,10=goal
    // obs = 0, rob = 50,  unknow = 100, vel = 150,  goal = 200, free = 250
    for (int m = 0; m < image_size; m++)
    {   
        state_msg_.laser_image.data[m] = 25*local_map_.data_[m];
        
    }
    image_pub_.publish(state_msg_.laser_image);
    publishMap(local_map_ ,state_msg_.laser_image);


    state_pub_.publish(state_msg_);
}

void ScanImage::goalCallback(const geometry_msgs::PoseStamped &msg)
{
    if (msg.pose.position.z < 0)
        has_goal_ = false;
    else
    {
        goal_frame_ = msg.header.frame_id;
        tf_goal_world_.setOrigin(tf::Vector3(msg.pose.position.x, msg.pose.position.y, 0));
        tf_goal_world_.setRotation(tf::Quaternion(msg.pose.orientation.x, msg.pose.orientation.y,
                                                  msg.pose.orientation.z, msg.pose.orientation.w));

        if (env_type_ == "real")
        {
            draw_goal_ = false;
            draw_vw_ = false;
        }
        else
        {
            if (msg.pose.position.z >= 1.0)
            {
                draw_goal_ = false;
                draw_vw_ = false;
            }
            else
            {
                draw_goal_ = true;
                draw_vw_ = false; // always shutdown
            }
        }
        has_goal_ = true;
    }
}

int ScanImage::publishMap(LocalMap &map, sensor_msgs::Image &image)
{
    nav_msgs::OccupancyGrid map_msg;
    map_msg.header.frame_id = track_frame_;
    map_msg.header.stamp = ros::Time::now();
    map_msg.info.height = map.size_x_;
    map_msg.info.width = map.size_y_;
    map_msg.info.origin.position.x = map.tf_map_base_.getOrigin().getX();
    map_msg.info.origin.position.y = map.tf_map_base_.getOrigin().getY();
    map_msg.info.origin.orientation.w = map.tf_map_base_.getRotation().getW();
    map_msg.info.origin.orientation.x = map.tf_map_base_.getRotation().getX();
    map_msg.info.origin.orientation.y = map.tf_map_base_.getRotation().getY();
    map_msg.info.origin.orientation.z = map.tf_map_base_.getRotation().getZ();
    map_msg.info.resolution = map.delta_;
    map_msg.data.clear();
    int j = 0;
    for (int i = 0; i < map.size_x_ * map.size_y_; i++)
    {
        int data = image.data[i];
        if (data == 125)//unknow
            map_msg.data.push_back(-1);
        else
            map_msg.data.push_back(255-data);
    }
    map_publisher_.publish(map_msg);
    return j;
}

bool ScanImage::checkGoalSrv(comn_pkg::CheckGoal::Request &req,comn_pkg::CheckGoal::Response &res) {

    tf::Stamped<tf::Pose> check_robot_world,check_goal_world_;
    if (env_type_ == "gazebo")
    {
        gazebo_msgs::GetModelState srv;
        srv.request.model_name = robot_name_;
        if(client_.call(srv) && srv.response.success)
        {
            geometry_msgs::Pose robot_pose = srv.response.pose;
            check_robot_world.setOrigin(tf::Vector3(robot_pose.position.x, robot_pose.position.y, 0));
            check_robot_world.setRotation(tf::Quaternion(robot_pose.orientation.x, robot_pose.orientation.y,
                                                      robot_pose.orientation.z, robot_pose.orientation.w));
        }
        else
            return false;
    }

    check_goal_world_.setOrigin(tf::Vector3(req.pose.x, req.pose.y, 0));
    check_goal_world_.setRotation(tf::Quaternion(0, 0, 0, 1));

    tf::Transform tf_check_goal_robot = check_robot_world.inverse() * check_goal_world_;

    float goal_x = tf_check_goal_robot.getOrigin().getX();
    float goal_y = tf_check_goal_robot.getOrigin().getY();
    //ROS_INFO("goal_x=%lf,goal_y=%lf",goal_x,goal_y);
    IntPoint p_goal;
    local_map_.getIndex(goal_x, goal_y, p_goal.x, p_goal.y);
    if(req.pose.z == 1)
    {
        res.result = local_map_.check_goal_1_m(p_goal.x,p_goal.y);
    }else
    {
        res.result = local_map_.check_goal_1_2_m(p_goal.x,p_goal.y);
    }

    //ROS_INFO("check goal x=%lf,y=%lf,result=%d,to x=%d, to y=%d",req.pose.x,req.pose.y,res.result,p_goal.x,p_goal.y);
    return true;

}

ScanImage::~ScanImage()
{
    if (syn_ != NULL)
    {
        delete syn_;
        syn_ = NULL;
    }
    if (scanSubscriber_ != NULL)
    {
        delete scanSubscriber_;
        scanSubscriber_ = NULL;
    }
    if (odomSubscriber_ != NULL)
    {
        delete odomSubscriber_;
        odomSubscriber_ = NULL;
    }
    for (int i = 0; i < thread_total_; i++)
    {
        thread_status_[i] = -1;
        delete thread_vector_[i];
        thread_vector_[i] = NULL;
    }
}


