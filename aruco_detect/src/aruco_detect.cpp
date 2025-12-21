#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <algorithm>
#include <map>

#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "fiducial_msgs/msg/fiducial.hpp"
#include "fiducial_msgs/msg/fiducial_array.hpp"
#include "fiducial_msgs/msg/fiducial_transform.hpp"
#include "fiducial_msgs/msg/fiducial_transform_array.hpp"

#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

#include <list>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/shared_ptr.hpp>

using namespace std;
using namespace cv;

typedef std::shared_ptr< fiducial_msgs::msg::FiducialArray const> FiducialArrayConstPtr;

class FiducialsNode : public rclcpp::Node {
  private:
    image_transport::Subscriber img_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr ignore_sub;
    rclcpp::Subscription<fiducial_msgs::msg::FiducialArray>::SharedPtr vertices_sub;

    rclcpp::Publisher<fiducial_msgs::msg::FiducialArray>::SharedPtr vertices_pub;
    rclcpp::Publisher<fiducial_msgs::msg::FiducialTransformArray>::SharedPtr pose_pub_fta;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pose_pub_d2a;
    image_transport::Publisher image_pub;

    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr service_enable_detections;

    tf2_ros::TransformBroadcaster broadcaster;

    bool publish_images;
    bool enable_detections;
    bool vis_msgs;
    bool verbose;

    double fiducial_len;

    bool doPoseEstimation;
    bool haveCamInfo;
    bool publishFiducialTf;
    vector <vector <Point2f> > corners;
    vector <int> ids;
    
    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    int frameNum;
    std::string frameId;
    std::vector<int> ignoreIds;
    std::map<int, double> fiducialLens;

    cv::Ptr<aruco::DetectorParameters> detectorParams;
    cv::Ptr<aruco::Dictionary> dictionary;

    void handleIgnoreString(const std::string& str);
    void estimatePoseSingleMarkers(float markerLength, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, vector<Vec3d>& rvecs, vector<Vec3d>& tvecs, vector<double>& reprojectionError);
    void ignoreCallback(const std_msgs::msg::String &msg);
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
    void poseEstimateCallback(const FiducialArrayConstPtr &msg);
    void camInfoCallback(const std::shared_ptr<const sensor_msgs::msg::CameraInfo> msg);
    bool enableDetectionsCallback(const std::shared_ptr<std_srvs::srv::SetBool::Request> request, std::shared_ptr<std_srvs::srv::SetBool::Response> response);

  public:
    FiducialsNode();
    void init();
};

/** Helper Functions **/
static void getSingleMarkerObjectPoints(float markerLength, vector<Point3f>& objPoints) {
    CV_Assert(markerLength > 0);
    objPoints.clear();
    objPoints.push_back(Vec3f(-markerLength / 2.f, markerLength / 2.f, 0));
    objPoints.push_back(Vec3f( markerLength / 2.f, markerLength / 2.f, 0));
    objPoints.push_back(Vec3f( markerLength / 2.f,-markerLength / 2.f, 0));
    objPoints.push_back(Vec3f(-markerLength / 2.f,-markerLength / 2.f, 0));
}

static double dist(const cv::Point2f &p1, const cv::Point2f &p2) {
    double dx = p1.x - p2.x; double dy = p1.y - p2.y;
    return sqrt(dx*dx + dy*dy);
}

static double getReprojectionError(const vector<Point3f> &objectPoints, const vector<Point2f> &imagePoints, const Mat &cameraMatrix, const Mat  &distCoeffs, const Vec3d &rvec, const Vec3d &tvec) {
    vector<Point2f> projectedPoints;
    try {
        cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
    } catch (...) { return 999.9; }
    
    double totalError = 0.0;
    for (unsigned int i=0; i<objectPoints.size() && i<projectedPoints.size(); i++) {
        double error = dist(imagePoints[i], projectedPoints[i]);
        totalError += error*error;
    }
    return totalError/(double)objectPoints.size();
}

/** Class Implementation **/
void FiducialsNode::estimatePoseSingleMarkers(float markerLength, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, vector<Vec3d>& rvecs, vector<Vec3d>& tvecs, vector<double>& reprojectionError) {
    CV_Assert(markerLength > 0);
    vector<Point3f> markerObjPoints;
    int nMarkers = (int)corners.size();
    rvecs.resize(nMarkers); tvecs.resize(nMarkers); reprojectionError.resize(nMarkers);
    for (int i = 0; i < nMarkers; i++) {
       double fiducialSize = markerLength;
       std::map<int, double>::iterator it = fiducialLens.find(ids[i]);
       if (it != fiducialLens.end()) { fiducialSize = it->second; }
       getSingleMarkerObjectPoints(fiducialSize, markerObjPoints);
       try {
           cv::solvePnP(markerObjPoints, corners[i], cameraMatrix, distCoeffs, rvecs[i], tvecs[i]);
           reprojectionError[i] = getReprojectionError(markerObjPoints, corners[i], cameraMatrix, distCoeffs, rvecs[i], tvecs[i]);
       } catch (...) {
           rvecs[i] = Vec3d(0,0,0); tvecs[i] = Vec3d(0,0,0);
       }
    }
}

void FiducialsNode::ignoreCallback(const std_msgs::msg::String& msg) {
    ignoreIds.clear();
    this->set_parameter(rclcpp::Parameter("ignore_fiducials", msg.data));
    handleIgnoreString(msg.data);
}

void FiducialsNode::camInfoCallback(const std::shared_ptr<const sensor_msgs::msg::CameraInfo> msg) {
    if (haveCamInfo) return;
    if (msg->k.size() >= 9) {
        cv::Mat tempMat = cv::Mat::eye(3, 3, CV_64F);
        for (int i=0; i<9; i++) tempMat.at<double>(i/3, i%3) = msg->k[i];
        if (tempMat.at<double>(0,0) == 0) return;
        cameraMatrix = tempMat.clone();
        if (msg->d.size() > 0) {
            distortionCoeffs = cv::Mat::zeros(1, (int)msg->d.size(), CV_64F);
            for (int i=0; i<(int)msg->d.size(); i++) distortionCoeffs.at<double>(0, i) = msg->d[i];
        } else {
            distortionCoeffs = cv::Mat::zeros(1, 5, CV_64F);
        }
        haveCamInfo = true;
        frameId = msg->header.frame_id;
        RCLCPP_INFO(this->get_logger(), "Camera intrinsics received: %dx%d", (int)msg->width, (int)msg->height);
    }
}

void FiducialsNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (!enable_detections) return;
    try {
        // --- แก้ไขแบบจบๆ: เลิกใช้ cv_bridge แปลง Auto แล้วสร้าง Mat จาก Raw Data เอง ---
        cv::Mat image;
        if (msg->encoding == "rgb8") {
            // Reconstruct จาก Raw data โดยกำหนด Step เองให้ถูกต้อง
            cv::Mat rgb_mat(msg->height, msg->width, CV_8UC3, const_cast<uchar*>(&msg->data[0]), msg->step);
            cv::cvtColor(rgb_mat, image, cv::COLOR_RGB2BGR);
        } else if (msg->encoding == "bgr8") {
            cv::Mat bgr_mat(msg->height, msg->width, CV_8UC3, const_cast<uchar*>(&msg->data[0]), msg->step);
            image = bgr_mat.clone(); // บังคับให้เป็น Continuous memory
        } else {
            // Fallback กรณี encoding อื่น ให้ลองใช้ cv_bridge (แต่ส่วนใหญ่ Realsense คือ rgb8)
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
            image = cv_ptr->image.clone();
        }

        if (image.empty()) return;

        corners.clear();
        ids.clear();
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams);

        fiducial_msgs::msg::FiducialArray fva;
        fva.header = msg->header;

        if(ids.size() > 0) {
            if(verbose) RCLCPP_INFO(this->get_logger(), "Detected %d markers", (int)ids.size());
            for (size_t i=0; i<ids.size(); i++) {
                if (std::count(ignoreIds.begin(), ignoreIds.end(), ids[i]) != 0) continue;
                fiducial_msgs::msg::Fiducial fid;
                fid.fiducial_id = ids[i];
                fid.x0 = corners[i][0].x; fid.y0 = corners[i][0].y;
                fid.x1 = corners[i][1].x; fid.y1 = corners[i][1].y;
                fid.x2 = corners[i][2].x; fid.y2 = corners[i][2].y;
                fid.x3 = corners[i][3].x; fid.y3 = corners[i][3].y;
                fva.fiducials.push_back(fid);
            }
            vertices_pub->publish(fva);
            
            if (publish_images) {
                aruco::drawDetectedMarkers(image, corners, ids);
                sensor_msgs::msg::Image::SharedPtr out_msg = cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
                image_pub.publish(out_msg);
            }
        } else if (publish_images) {
            // ส่งภาพเปล่าๆ ออกไปถ้าไม่เจอ marker (ใช้ cv_bridge ได้เพราะไม่ได้ประมวลผลต่อ)
            sensor_msgs::msg::Image::SharedPtr out_msg = cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
            image_pub.publish(out_msg);
        }
    } catch(const std::exception & e) { 
        RCLCPP_ERROR(this->get_logger(), "Error in imageCallback: %s", e.what()); 
    }
}

void FiducialsNode::poseEstimateCallback(const FiducialArrayConstPtr & msg) {
    if (!doPoseEstimation || !haveCamInfo || ids.empty() || cameraMatrix.empty()) return;
    try {
        vector <Vec3d> rvecs, tvecs;
        vector <double> reprojectionError;
        estimatePoseSingleMarkers((float)fiducial_len, cameraMatrix, distortionCoeffs, rvecs, tvecs, reprojectionError);

        fiducial_msgs::msg::FiducialTransformArray fta;
        fta.header.stamp = msg->header.stamp;
        fta.header.frame_id = frameId;

        for (size_t i=0; i<ids.size(); i++) {
            double angle = norm(rvecs[i]);
            if (angle < 0.00001) continue; 
            Vec3d axis = rvecs[i] / angle;
            tf2::Quaternion q;
            q.setRotation(tf2::Vector3(axis[0], axis[1], axis[2]), angle);

            fiducial_msgs::msg::FiducialTransform ft;
            ft.fiducial_id = ids[i];
            ft.transform.translation.x = tvecs[i][0]; ft.transform.translation.y = tvecs[i][1]; ft.transform.translation.z = tvecs[i][2];
            ft.transform.rotation.w = q.w(); ft.transform.rotation.x = q.x(); ft.transform.rotation.y = q.y(); ft.transform.rotation.z = q.z();
            fta.transforms.push_back(ft);

            if (publishFiducialTf) {
                geometry_msgs::msg::TransformStamped ts;
                ts.header.stamp = msg->header.stamp;
                ts.header.frame_id = frameId;
                ts.child_frame_id = "fiducial_" + std::to_string(ids[i]);
                ts.transform.translation.x = tvecs[i][0]; ts.transform.translation.y = tvecs[i][1]; ts.transform.translation.z = tvecs[i][2];
                ts.transform.rotation.w = q.w(); ts.transform.rotation.x = q.x(); ts.transform.rotation.y = q.y(); ts.transform.rotation.z = q.z();
                broadcaster.sendTransform(ts);
            }
        }
        pose_pub_fta->publish(fta);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in poseEstimateCallback: %s", e.what());
    }
}

void FiducialsNode::handleIgnoreString(const std::string& str) {
    std::vector<std::string> strs; boost::split(strs, str, boost::is_any_of(","));
    for (const string& element : strs) {
        if (element == "") continue;
        try {
            std::vector<std::string> range; boost::split(range, element, boost::is_any_of("-"));
            if (range.size() == 2) {
               for (int j=std::stoi(range[0]); j<=std::stoi(range[1]); j++) ignoreIds.push_back(j);
            } else if (range.size() == 1) { ignoreIds.push_back(std::stoi(range[0])); }
        } catch (...) {}
    }
}

bool FiducialsNode::enableDetectionsCallback(const std::shared_ptr<std_srvs::srv::SetBool::Request> request, std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
    enable_detections = request->data;
    response->success = true;
    return true;
}

FiducialsNode::FiducialsNode() : Node("aruco_detect"), broadcaster(this) {
    haveCamInfo = false; 
    enable_detections = true;
    detectorParams = cv::makePtr<aruco::DetectorParameters>();
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distortionCoeffs = cv::Mat::zeros(1, 5, CV_64F);

    this->declare_parameter("publish_images", true);
    this->declare_parameter("fiducial_len", 0.14);
    this->declare_parameter("dictionary", 0);
    this->declare_parameter("do_pose_estimation", true);
    this->declare_parameter("publish_fiducial_tf", true);
    this->declare_parameter("verbose", true);
    this->declare_parameter("ignore_fiducials", "");

    this->get_parameter("publish_images", publish_images);
    this->get_parameter("fiducial_len", fiducial_len);
    this->get_parameter("do_pose_estimation", doPoseEstimation);
    this->get_parameter("publish_fiducial_tf", publishFiducialTf);
    this->get_parameter("verbose", verbose);
    
    int dicno; this->get_parameter("dictionary", dicno);
    dictionary = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(dicno));

    vertices_pub = this->create_publisher<fiducial_msgs::msg::FiducialArray>("fiducial_vertices", 10);
    pose_pub_fta = this->create_publisher<fiducial_msgs::msg::FiducialTransformArray>("fiducial_transforms", 10);
    image_pub = image_transport::create_publisher(this, "fiducial_images");

    vertices_sub = this->create_subscription<fiducial_msgs::msg::FiducialArray>("fiducial_vertices", 10, std::bind(&FiducialsNode::poseEstimateCallback, this, std::placeholders::_1));
    caminfo_sub = this->create_subscription<sensor_msgs::msg::CameraInfo>("camera_info", rclcpp::SensorDataQoS(), std::bind(&FiducialsNode::camInfoCallback, this, std::placeholders::_1));
    service_enable_detections = this->create_service<std_srvs::srv::SetBool>("enable_detections", std::bind(&FiducialsNode::enableDetectionsCallback, this, std::placeholders::_1, std::placeholders::_2));
}

void FiducialsNode::init(){
    rmw_qos_profile_t qos = rmw_qos_profile_sensor_data;
    image_transport::ImageTransport it(this->shared_from_this());
    img_sub = it.subscribe("image", qos, &FiducialsNode::imageCallback, this);
    RCLCPP_INFO(this->get_logger(), "Aruco Detect Initialized.");
}

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FiducialsNode>();
    node->init();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}