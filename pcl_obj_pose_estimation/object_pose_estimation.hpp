
/* 
 *  CPP Header file for 'ObjectPoseEstimate2D' class
 * 
 *  Created By: Tan You Liang, Feb 2019
 *  - for testing on 2D pose estimation of targeted object (line)
 * 
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


#include <Eigen/Dense>

#define PI 3.14159265


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
    std::vector<Eigen::Vector3f> *targetPoseArray;

    Eigen::Vector3f TargetPose;
    int target_line_idx;  // final identified targeted line's idx
    int jump_count; // for jump filtering
    
    // Param stuffs
    YAML::Node config;
    Eigen::Vector4f roi_range; // Region of interest [x.min, x.max, y.min, y.max]
    float target_length, length_tolerance;
    bool enable_outliner_filtering;
    float outliner_mean_k, dist_coeff_factor;
    float ransac_dist_thresh, outliner_std_dev_factor; 
    int min_num_points, averaging_span,jump_count_allowance;
    float jump_score_thresh;


  protected:    

    void objectClustering();

    void lineFitting();

    // called by line fitting    
    void getLinesDescriptors(
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Eigen::VectorXf coeff);

    void findTargetPose();


  public:

    // @arg: input path for yaml config file
    ObjectPoseEstimate2D(std::string config_path);
    
    // @arg: input cloud for 2d line detection
    void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    // use when would like to avg the result, smoothern the curve , 
    // uses in multiple realtime inputs
    // @param: tune with 'averaging_span, jump val' in config
    void applyMovingAvgFiltering();

    // get Eigen[x, y, theta(rad)] of targeted obj pose
    void getTargetPose( Eigen::Vector3f *target_pose);

    // get target object point cloud line
    void getTargetPointCloud( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    // only for pcl visualization
    pcl::visualization::PCLVisualizer::Ptr simpleVis ();

    // re initiate same class
    void reInit();

    // get region of interest
    void getROI(std::vector<Eigen::Vector3f> *roi_points);

    // TODO:
    // getLinesPoints();
    // getROI();
};
