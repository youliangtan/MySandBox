
/* 
 *  Created By: Tan You Liang, Feb 2019
 *  - for testing on ransac interested object identification and pose estimation
 *  - Created for Testing
*/

#include <iostream>
#include <string>
#include <math.h>    
#include <yaml-cpp/yaml.h>



#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>


// clustering and filtering
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>


#include <boost/thread/thread.hpp>
#include <Eigen/Dense>


#define PI 3.14159265
#define TARGET_LENGTH 0.55
#define LENGTH_TOLERANCE 0.08


// TODO: struct line(length, pose, numPoints, distance)
struct LineDescriptor {
  int index;
  int num_points;
  float distance;
  float length;
  float x_max;
  float y_max;
  float x_min;
  float y_min; 
  float mid_x;
  float mid_y;
  float theta;
};


// PCL Object Pose Estimation
class ObjectPoseEstimate2D {
  private:
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> *clusters_cloud;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> *lines_cloud;
    std::vector<LineDescriptor> *lines_descriptors;
    Eigen::Vector3f TargetPose;
    int target_line_idx;

    
    // TODO: Param stuffs
    YAML::Node config;
    Eigen::Vector4f roi_range; // Region of interest [x.min, x.max, y.min, y.max]
    float target_length, length_tolerance, min_num_points;
    bool enable_outliner_filtering;
    float outliner_mean_k, dist_coeff_factor, ransac_dist_thresh, outliner_std_dev_factor; //line_fitting


  protected:    

    void objectClustering();

    void lineFitting();

    // called by line fitting    
    void getLinesDescriptors(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Eigen::VectorXf coeff);


  public:

    ObjectPoseEstimate2D(std::string config_path);
    
    void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    // Eigen[x, y, theta]
    void getTargetPose( Eigen::Vector3f *target_pose);

    // get target point cloud line
    void getTargetPointCloud( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    // for visualization
    pcl::visualization::PCLVisualizer::Ptr simpleVis ();

    // TODO:
    // getLinesPoints();
    // getROI();
    // getLinePoints();

};
