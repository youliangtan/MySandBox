
/* 
 *  Created By: Tan You Liang, Feb 2019
 *  - for testing on ransac interested object identification and pose estimation
 *  - Created for Testing
*/

#include "object_pose_estimation.h"



  
// Init
ObjectPoseEstimate2D::ObjectPoseEstimate2D(std::string config_path){

  std::cout << "Starting PCL Object Pose Estimation" << std::endl;
  input_cloud = (boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> >) new pcl::PointCloud<pcl::PointXYZ>() ;
  clusters_cloud = new std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>;
  lines_cloud = new std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>;
  lines_descriptors = new std::vector<LineDescriptor>;

  // // ========= load param ============

  YAML::Node config = YAML::LoadFile(config_path);

  // Create PCL Clustering init
  ec.setClusterTolerance ( config["cluster_tolerance"].as<float>() );
  ec.setMinClusterSize ( config["min_cluster_size"].as<float>() );
  ec.setMaxClusterSize ( config["max_cluster_size"].as<float>() );

  // line fitting
  enable_outliner_filtering = config["enable_outliner_filtering"].as<bool>();
  outliner_mean_k = config["outliner_mean_k"].as<float>();
  dist_coeff_factor = config["cluster_tolerance"].as<float>();
  ransac_dist_thresh = config["ransac_dist_thresh"].as<float>();
  outliner_std_dev_factor = config["outliner_std_dev_factor"].as<float>();

  // target identification
  roi_range[0] = config["region_of_interest"]["x_min"].as<float>(); // [x.min, x.max, y.min, y.max]
  roi_range[1] = config["region_of_interest"]["x_max"].as<float>();;
  roi_range[2] = config["region_of_interest"]["y_min"].as<float>();;
  roi_range[3] = config["region_of_interest"]["y_max"].as<float>();;
  target_length = config["target_length"].as<float>();
  length_tolerance = config["length_tolerance"].as<float>(); 
  min_num_points = config["min_num_points"].as<int>();

}


// PCL Visualizer
pcl::visualization::PCLVisualizer::Ptr ObjectPoseEstimate2D::simpleVis (){

  std::cout << " Start PCL Visualizer" << std::endl;


  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->addCoordinateSystem (1.0, "axis");
  viewer->setBackgroundColor (0.01, 0.01, 0.01);

  // Original cloud
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (input_cloud, 0, 0, 255);
  viewer->addPointCloud<pcl::PointXYZ> (input_cloud, source_cloud_color_handler, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  


  // visualizting clusters_cloud
  int cluster_idx=0;
  for(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>::iterator points = clusters_cloud->begin(); points != clusters_cloud->end(); ++points) {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler2 (*points, 0, 255, 0);
    std::string idx = std::to_string(cluster_idx); //convert int to str
    viewer->addPointCloud<pcl::PointXYZ> (*points, source_cloud_color_handler2, "cluster cloud" + idx);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cluster cloud" + idx);
    cluster_idx++;
  }
  

  // visualizting all lines points
  int line_idx=0;
  for(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>::iterator points = lines_cloud->begin(); points != lines_cloud->end(); ++points) {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler3 (*points, 255, 0, 0);
    std::string idx = std::to_string(line_idx); //convert int to str
    viewer->addPointCloud<pcl::PointXYZ> (*points, source_cloud_color_handler3, "line cloud" + idx);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "line cloud" + idx);
    line_idx++;
  }


  // Visualize line descriptors
  for (int idx = 0; idx < lines_descriptors->size(); ++idx){
    // text
    pcl::PointXYZ line_center;
    line_center.x = lines_descriptors->at(idx).mid_x;
    line_center.y = lines_descriptors->at(idx).mid_y;
    line_center.z = 0;
    viewer->addText3D(  "  line" + std::to_string(idx) + "\n  length : " + 
                        std::to_string( lines_descriptors->at(idx).length ) + "\n  theta: " +
                        std::to_string( lines_descriptors->at(idx).theta ), 
                        line_center, 0.04, 0.0, 1.0, 0.0, 
                        "line"+ std::to_string(idx));

    // 1 line 2 endpoints
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_points (new pcl::PointCloud<pcl::PointXYZ>);
    line_points->width  = 2;
    line_points->height = 1;
    line_points->points.resize (line_points->width * line_points->height);
    line_points->points[0].x = lines_descriptors->at(idx).x_min;
    line_points->points[0].y = lines_descriptors->at(idx).y_min;
    line_points->points[0].z = 0;
    line_points->points[1].x = lines_descriptors->at(idx).x_max;
    line_points->points[1].y = lines_descriptors->at(idx).y_max;
    line_points->points[1].z = 0;
    
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler4 (line_points, 255, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ> (line_points, source_cloud_color_handler4, "endpoints" + std::to_string(idx));
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "endpoints" + std::to_string(idx));
  }


  // Visualize Target Pose Arrow
  pcl::PointXYZ start_point, end_point;
  float rad = TargetPose[2]*PI/180; // deg to rad
  start_point.x = TargetPose[0];
  start_point.y = TargetPose[1];
  start_point.z = 0;
  end_point.x = TargetPose[0]-0.5*sin(rad);
  end_point.y = TargetPose[1]+0.5*cos(rad);
  end_point.z = 0;
  viewer->addArrow (end_point, start_point, 1, 1, 1, "target_arrow");


  viewer->initCameraParameters ();
  viewer->setCameraPosition(0,0,6,0,0,0);

  return (viewer);
}



// identify all line's endpoints
// 2D plane only
// return pose
void ObjectPoseEstimate2D::getLinesDescriptors(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Eigen::VectorXf coeff){
  
  //  ===================== get_line_endpoints ======================
  // init
  float score, score_max, score_min;
  score = cloud->points[0].x * coeff[3]  + cloud->points[0].y * coeff[4];
  score_max = score, 
  score_min = score;
  float x_min, y_min, x_max, y_max;
  x_min = cloud->points[0].x;
  y_min = cloud->points[0].y;
  x_max = cloud->points[0].x;
  y_max = cloud->points[0].y;

  for (int i = 1; i < cloud->points.size (); ++i){
    score = (cloud->points[i].x * coeff[3]) + (cloud->points[i].y * coeff[4]);
    // std::cout << "point :" << cloud->points[i].x * coeff[3] << " "  << cloud->points[i].y << std::endl;  
    // std::cout << "Score :" << score << " \t| x, Min, Max " << x_min << " " << x_max << std::endl;  

    if ( score > score_max ){
      score_max = score;
      x_max = cloud->points[i].x;
      y_max = cloud->points[i].y;
    }

    if ( score < score_min ){
      score_min = score;
      x_min = cloud->points[i].x;
      y_min = cloud->points[i].y;
    }
  }

  // std::cout << "Score {min, max}: " << score_min << " " << score_max << std::endl;  
  // std::cout << "Min {x, y} :" << x_min << " " << y_min << std::endl;
  // std::cout << "Max {x, y} :" << x_max << " " << y_max << std::endl;

  float length; //todo remove sq rt
  length = sqrt( (x_max - x_min)*(x_max - x_min) + (y_max - y_min)*(y_max - y_min) );
  std::cout << " - length: " <<  length <<std::endl;


  // ========================= cont' with line profiling ==============================

  struct LineDescriptor line_desc;
  line_desc.index = lines_descriptors->size();
  line_desc.num_points = cloud->points.size ();
  line_desc.x_max = x_max;
  line_desc.y_max = y_max;
  line_desc.x_min = x_min;
  line_desc.y_min = y_min;
  line_desc.mid_x = (x_max + x_min )/2;
  line_desc.mid_y = (y_max + y_min )/2;
  line_desc.length = sqrt( (x_max - x_min)*(x_max - x_min) + (y_max - y_min)*(y_max - y_min) );
  line_desc.distance = sqrt( (line_desc.mid_x)*(line_desc.mid_x) + (line_desc.mid_y)*(line_desc.mid_y) );
  line_desc.theta = atan (coeff[4]/coeff[3]) * 180 / PI;    // degrees

  // update line description list datas
  lines_descriptors->push_back ( line_desc );
  
}




// clustering and filtering
void ObjectPoseEstimate2D::objectClustering(){
  
  // pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

  // clusterize each plane
  tree->setInputCloud (input_cloud); //TODO check if theres any use of kd tree func
  ec.setSearchMethod (tree);
  std::vector<pcl::PointIndices> cluster_indices;
  ec.setInputCloud (input_cloud);
  ec.extract (cluster_indices);

  std::cout << "Num of Clusters: " << cluster_indices.size () << std::endl;

  // extract and visualize cluster segmentation for each plane
  int clusterNum = 0;

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    //create cluster
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (input_cloud->points[*pit]);
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    // std::cout << "\n- PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;

    // update to list of clusters
    clusters_cloud->push_back ( cloud_cluster );  
    clusterNum ++;
  }

}



// Ransac Line Fitting
void ObjectPoseEstimate2D::lineFitting(){
  
  float dist_coeff;

  for(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>::iterator cloud = clusters_cloud->begin(); cloud != clusters_cloud->end(); ++cloud) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr target (new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> inliers;
    Eigen::VectorXf coeff;  //  * [point_on_line.x point_on_line.y point_on_line.z line_direction.x line_direction.y line_direction.z] (unit vector)

    //ransac
    pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr model_l (new pcl::SampleConsensusModelLine<pcl::PointXYZ> (*cloud));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_l);
    ransac.setDistanceThreshold (ransac_dist_thresh); // so call error allowance for laser scan
    ransac.computeModel();
    ransac.getInliers(inliers);
    ransac.getModelCoefficients(coeff);

    std::cout << "Line coeff: " << coeff[0] << " " << coeff[1] << " " << coeff[3] << " " << coeff[4] << std::endl;

    /// find lines' end points
    pcl::copyPointCloud<pcl::PointXYZ>(**cloud, inliers, *target);
    
    // // outliner filtering
    if (enable_outliner_filtering){
      dist_coeff = (coeff[0]*coeff[0] + coeff[1]*coeff[1])* dist_coeff_factor; // 0.1 is approx
      std::cout << "Distance coeff: " << dist_coeff << std::endl;
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud (target);
      sor.setMeanK (outliner_mean_k);
      sor.setStddevMulThresh ( outliner_std_dev_factor * dist_coeff);
      sor.filter (*target);
    }

    
    getLinesDescriptors(target, coeff);
    lines_cloud->push_back ( target );   // pointcloud

    std::cout << "- PointCloud representing the line: " << target->points.size () << " data points. \n" << std::endl;

  }

}
  


// get pose estimation
// param input: roi_range[4], target_length, length_tolerance, min_num_points
void ObjectPoseEstimate2D::getTargetPose( Eigen::Vector3f *target_pose ){

  float mid_x, mid_y, x_coor, y_coor, theta, length;

  target_line_idx = -1;    
  x_coor = 0;
  y_coor = 0;
  theta = 0;


  std::cout<< "Total set of lines: " << lines_descriptors->size() << std::endl;

  for (int idx = 0; idx < lines_descriptors->size(); ++idx){
    
    // check x_y range 
    mid_x = lines_descriptors->at(idx).mid_x;
    mid_y = lines_descriptors->at(idx).mid_y;

    if (mid_x < roi_range[0] || mid_x > roi_range[1]) continue;
    if (mid_y < roi_range[2] || mid_y > roi_range[3]) continue;

    // length with tolerance
    length = lines_descriptors->at(idx).length;
    if ( length > (target_length + length_tolerance) || length < (target_length - length_tolerance) ) continue;

    // check num of points
    if ( lines_descriptors->at(idx).num_points < min_num_points) continue;

    // comfirm that pass
    std::cout<< "FOUND Target at index: " << idx << std::endl;
    target_line_idx = idx;
    x_coor = mid_x;
    y_coor = mid_y;
    theta = lines_descriptors->at(idx).theta;

    break;
  }

  (*target_pose)[0] = x_coor;
  (*target_pose)[1] = y_coor;
  (*target_pose)[2] = theta;
  
  TargetPose = *target_pose;
  
}



void ObjectPoseEstimate2D::getTargetPointCloud( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  if (target_line_idx != -1) *cloud = *(lines_cloud->at(target_line_idx));
}



void ObjectPoseEstimate2D::setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  input_cloud = cloud;
  std::cout << "[Obj Pose Estimation]::Num of Input Points: " << input_cloud->size () << "\n"<< std::endl;
  // pcl processing
  objectClustering();
  lineFitting();
}






//////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// ------------------ Main Function ------------------- //////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char** argv)
{
  // initialize PointClouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr target (new pcl::PointCloud<pcl::PointXYZ>); 
  Eigen::Vector3f target_pose;


  if (argc < 3 || pcl::console::find_switch (argc, argv, "-h") )
  {
      cout<<" - Run this script to process .pcd file to ransac seg "<<endl;
      cout<<"Usage: ./random_sample_consensus -l -input <input_file>"<<endl;
      return -1;
  }

  //get arg pcdinput 
  if (pcl::console::find_argument (argc, argv, "-input") >= 0){
      int input_idx = pcl::console::find_argument (argc, argv, "-input") + 1;
      std::string input_file = argv[input_idx];
      std::cout<<"File Path: "<<input_file<<std::endl;
      pcl::io::loadPCDFile<pcl::PointXYZ> ( input_file, *cloud );
      std::cout<<"Point cloud loaded, point size = "<<cloud->points.size()<<std::endl;
  } 
  else{
    std::cout<<"No Input PCD File, pls input via '-input' "<<std::endl;
    exit(0);
  }


  // =============== Place class here =====================
  ObjectPoseEstimate2D agv_laser_scan(config_path="../config.yaml");
  agv_laser_scan.setInputCloud(cloud);
  agv_laser_scan.getTargetPose(&target_pose);
  agv_laser_scan.getTargetPointCloud(target);

  std::cout << " Target Object Pose: " << target_pose[0] << " " << target_pose[1] << " " << target_pose[2] << std::endl;
  std::cout << " - target size: " << target->size () << "\n"<< std::endl;


  // visualizer
  pcl::visualization::PCLVisualizer::Ptr viewer;
  viewer = agv_laser_scan.simpleVis();

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
  return 0;

 }
