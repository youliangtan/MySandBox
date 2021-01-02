
/* 
 * CPP code on PCL execution of 'ObjectPoseEstimate2D' class, 
 * 
 *  Created By: Tan You Liang, Feb 2019
 *  - Run 2D pose estimation of a pointcloud input to find a targeted object (line)
 *  - Utilized PCL lib in this class
 *  - In main, single lidar scan of a input .pcd file is given.
 *  - pcl simple visualization is used here (optional)
*/

#include "object_pose_estimation.hpp"


// Init
ObjectPoseEstimate2D::ObjectPoseEstimate2D(std::string config_path)
{
  std::cout << "Starting PCL Object Pose Estimation" << std::endl;
  input_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  clusters_cloud = new std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>;
  lines_cloud = new std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>;
  lines_descriptors = new std::vector<LineDescriptor>;
  targetPoseArray = new std::vector<Eigen::Vector3f>;

  // // ========= load param ============

  YAML::Node config = YAML::LoadFile(config_path);

  std::cout << "Yeah!!!! YAML is Loaded ... \n"
            << std::endl;

  // Create PCL Clustering init
  ec.setClusterTolerance(config["cluster_tolerance"].as<float>());
  ec.setMinClusterSize(config["min_cluster_size"].as<float>());
  ec.setMaxClusterSize(config["max_cluster_size"].as<float>());

  // line fitting
  enable_outliner_filtering = config["enable_outliner_filtering"].as<bool>();
  outliner_mean_k = config["outliner_mean_k"].as<float>();
  dist_coeff_factor = config["cluster_tolerance"].as<float>();
  ransac_dist_thresh = config["ransac_dist_thresh"].as<float>();
  outliner_std_dev_factor = config["outliner_std_dev_factor"].as<float>();

  // target identification
  /// [x.min, x.max, y.min, y.max]
  roi_range[0] = config["region_of_interest"]["x_min"].as<float>();
  roi_range[1] = config["region_of_interest"]["x_max"].as<float>();
  roi_range[2] = config["region_of_interest"]["y_min"].as<float>();
  roi_range[3] = config["region_of_interest"]["y_max"].as<float>();
  target_length = config["target_length"].as<float>();
  length_tolerance = config["length_tolerance"].as<float>();
  min_num_points = config["min_num_points"].as<int>();

  // averaging filtering
  averaging_span = config["averaging_span"].as<int>();
  jump_score_thresh = config["jump_score_thresh"].as<float>();
  jump_count_allowance = config["jump_count_allowance"].as<int>();
}

// re initialize for every scan
void ObjectPoseEstimate2D::reInit()
{
  input_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  clusters_cloud = new std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>;
  lines_cloud = new std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>;
  lines_descriptors = new std::vector<LineDescriptor>;
}

// PCL Visualizer, blocking while visualizing
pcl::visualization::PCLVisualizer::Ptr ObjectPoseEstimate2D::simpleVis()
{
  using namespace pcl::visualization;
  std::cout << " Start PCL Visualizer" << std::endl;

  PCLVisualizer::Ptr viewer(new PCLVisualizer("3D Viewer"));
  viewer->addCoordinateSystem(1.0, "axis");
  viewer->setBackgroundColor(0.01, 0.01, 0.01);

  // Original cloud
  PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler(input_cloud, 0, 0, 255);
  viewer->addPointCloud<pcl::PointXYZ>(input_cloud, source_cloud_color_handler, "sample cloud");
  viewer->setPointCloudRenderingProperties(PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");

  // visualizting clusters_cloud
  int cluster_idx = 0;
  for (auto points = clusters_cloud->begin(); points != clusters_cloud->end(); ++points)
  {
    PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler2(*points, 0, 255, 0);
    std::string idx = std::to_string(cluster_idx); //convert int to str
    viewer->addPointCloud<pcl::PointXYZ>(
      *points, source_cloud_color_handler2, "cluster cloud" + idx);
    viewer->setPointCloudRenderingProperties(
      PCL_VISUALIZER_POINT_SIZE, 4, "cluster cloud" + idx);
    cluster_idx++;
  }

  // visualizting all lines points
  int line_idx = 0;
  for (auto points = lines_cloud->begin(); points != lines_cloud->end(); ++points)
  {
    PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler3(*points, 255, 0, 0);
    std::string idx = std::to_string(line_idx); //convert int to str
    viewer->addPointCloud<pcl::PointXYZ>(*points, source_cloud_color_handler3, "line cloud" + idx);
    viewer->setPointCloudRenderingProperties(PCL_VISUALIZER_POINT_SIZE, 5, "line cloud" + idx);
    line_idx++;
  }

  // Visualize line descriptors
  for (int idx = 0; idx < lines_descriptors->size(); ++idx)
  {
    // text
    pcl::PointXYZ line_center;
    line_center.x = lines_descriptors->at(idx).mid_x;
    line_center.y = lines_descriptors->at(idx).mid_y;
    line_center.z = 0;
    viewer->addText3D(" line" + std::to_string(idx) + "\n  length : " +
                        std::to_string(lines_descriptors->at(idx).length) + 
                        "\n  theta(degrees): " +
                        std::to_string(lines_descriptors->at(idx).theta * 180 / PI),
                      line_center, 0.04, 0.0, 1.0, 0.0,
                      "line" + std::to_string(idx));

    // 1 line 2 endpoints
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_points(new pcl::PointCloud<pcl::PointXYZ>);
    line_points->width = 2;
    line_points->height = 1;
    line_points->points.resize(line_points->width * line_points->height);
    line_points->points[0].x = lines_descriptors->at(idx).x_min;
    line_points->points[0].y = lines_descriptors->at(idx).y_min;
    line_points->points[0].z = 0;
    line_points->points[1].x = lines_descriptors->at(idx).x_max;
    line_points->points[1].y = lines_descriptors->at(idx).y_max;
    line_points->points[1].z = 0;

    PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler4(
      line_points, 255, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(
      line_points, source_cloud_color_handler4, "endpoints" + std::to_string(idx));
    viewer->setPointCloudRenderingProperties(
      PCL_VISUALIZER_POINT_SIZE, 15, "endpoints" + std::to_string(idx));
  }

  // Visualize Target Pose Arrow
  pcl::PointXYZ start_point, end_point;
  float rad = TargetPose[2] * PI / 180; // deg to rad
  start_point.x = TargetPose[0];
  start_point.y = TargetPose[1];
  start_point.z = 0;
  end_point.x = TargetPose[0] - 0.5 * sin(rad);
  end_point.y = TargetPose[1] + 0.5 * cos(rad);
  end_point.z = 0;
  viewer->addArrow(end_point, start_point, 1, 1, 1, "target_arrow");

  viewer->initCameraParameters();
  viewer->setCameraPosition(0, 0, 6, 0, 0, 0);

  return (viewer);
}

// identify all line's endpoints, get line description for later evaluation
// for 2D plane only
// @return pose
void ObjectPoseEstimate2D::getLinesDescriptors(
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Eigen::VectorXf coeff)
{

  //  ===================== get_line_endpoints ======================
  // init
  float score, score_max, score_min;
  score = cloud->points[0].x * coeff[3] + cloud->points[0].y * coeff[4];
  score_max = score,
  score_min = score;
  float x_min, y_min, x_max, y_max;
  x_min = cloud->points[0].x;
  y_min = cloud->points[0].y;
  x_max = cloud->points[0].x;
  y_max = cloud->points[0].y;

  for (int i = 1; i < cloud->points.size(); ++i)
  {
    score = (cloud->points[i].x * coeff[3]) + (cloud->points[i].y * coeff[4]);
    // std::cout << "point :" << cloud->points[i].x * coeff[3] 
    //           << " "  << cloud->points[i].y << std::endl;
    // std::cout << "Score :" << score << " \t| x, Min, Max " 
    //           << x_min << " " << x_max << std::endl;

    if (score > score_max)
    {
      score_max = score;
      x_max = cloud->points[i].x;
      y_max = cloud->points[i].y;
    }

    if (score < score_min)
    {
      score_min = score;
      x_min = cloud->points[i].x;
      y_min = cloud->points[i].y;
    }
  }

  // std::cout << "Score {min, max}: " << score_min << " " << score_max << std::endl;
  // std::cout << "Min {x, y} :" << x_min << " " << y_min << std::endl;
  // std::cout << "Max {x, y} :" << x_max << " " << y_max << std::endl;

  float length; //todo remove sq rt
  length = sqrt((x_max - x_min) * (x_max - x_min) + (y_max - y_min) * (y_max - y_min));
  std::cout << " - length: " << length << std::endl;

  // ========================= cont' with line profiling ==============================

  struct LineDescriptor line_desc;
  line_desc.index = lines_descriptors->size();
  line_desc.num_points = cloud->points.size();
  line_desc.x_max = x_max;
  line_desc.y_max = y_max;
  line_desc.x_min = x_min;
  line_desc.y_min = y_min;
  line_desc.mid_x = (x_max + x_min) / 2;
  line_desc.mid_y = (y_max + y_min) / 2;
  line_desc.length = 
    sqrt((x_max - x_min) * (x_max - x_min) + (y_max - y_min) * (y_max - y_min));
  line_desc.distance = 
    sqrt((line_desc.mid_x) * (line_desc.mid_x) + (line_desc.mid_y) * (line_desc.mid_y));
  line_desc.theta = atan(coeff[4] / coeff[3]); // degrees

  // update line description list datas
  lines_descriptors->push_back(line_desc);
}

// clustering and filtering
void ObjectPoseEstimate2D::objectClustering()
{

  // pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

  // clusterize each plane
  tree->setInputCloud(input_cloud);
  ec.setSearchMethod(tree);
  std::vector<pcl::PointIndices> cluster_indices;
  ec.setInputCloud(input_cloud);
  ec.extract(cluster_indices);

  std::cout << "Num of Clusters: " << cluster_indices.size() << std::endl;

  // extract and visualize cluster segmentation for each plane
  int clusterNum = 0;

  for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
  {
    //create cluster
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto pit = it->indices.begin(); pit != it->indices.end(); ++pit)
      cloud_cluster->points.push_back(input_cloud->points[*pit]);
    cloud_cluster->width = cloud_cluster->points.size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    // update to list of clusters
    clusters_cloud->push_back(cloud_cluster);
    clusterNum++;
  }
}

// Ransac Line Fitting
void ObjectPoseEstimate2D::lineFitting()
{
  float dist_coeff;

  for (auto cloud = clusters_cloud->begin(); cloud != clusters_cloud->end(); ++cloud)
  {
    auto target = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    std::vector<int> inliers;
    /// [ point_on_line.x point_on_line.y 
    ///   point_on_line.z line_direction.x line_direction.y line_direction.z] (unit vector)
    Eigen::VectorXf coeff;

    //ransac
    auto model_l = boost::make_shared<pcl::SampleConsensusModelLine<pcl::PointXYZ>>(*cloud);
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_l);
    ransac.setDistanceThreshold(ransac_dist_thresh); // so call error allowance for laser scan
    ransac.computeModel();
    ransac.getInliers(inliers);
    ransac.getModelCoefficients(coeff);

    std::cout << "Line coeff: " << coeff[0] << " " << coeff[1] << " " 
              << coeff[3] << " " << coeff[4] << std::endl;

    /// find lines' end points
    pcl::copyPointCloud<pcl::PointXYZ>(**cloud, inliers, *target);

    // // outliner filtering
    if (enable_outliner_filtering)
    {
      dist_coeff = (coeff[0] * coeff[0] + coeff[1] * coeff[1]) * dist_coeff_factor; // 0.1 is approx
      std::cout << "Distance coeff: " << dist_coeff << std::endl;
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud(target);
      sor.setMeanK(outliner_mean_k);
      sor.setStddevMulThresh(outliner_std_dev_factor * dist_coeff);
      sor.filter(*target);
    }

    getLinesDescriptors(target, coeff);
    lines_cloud->push_back(target); // pointcloud

    std::cout << "- PointCloud representing the line: " 
              << target->points.size() << " data points. "  << std::endl;
  }
}

// find pose estimation of target
void ObjectPoseEstimate2D::findTargetPose()
{
  float mid_x, mid_y, x_coor, y_coor, theta, length;

  target_line_idx = -1;
  x_coor = 0;
  y_coor = 0;
  theta = 0;

  std::cout << "Total set of lines: " << lines_descriptors->size() << std::endl;

  for (int idx = 0; idx < lines_descriptors->size(); ++idx)
  {

    // check x_y range
    // param input: roi_range[4], target_length, length_tolerance, min_num_points
    mid_x = lines_descriptors->at(idx).mid_x;
    mid_y = lines_descriptors->at(idx).mid_y;

    if (mid_x < roi_range[0] || mid_x > roi_range[1])
      continue;
    if (mid_y < roi_range[2] || mid_y > roi_range[3])
      continue;

    // length with tolerance
    length = lines_descriptors->at(idx).length;
    if (length > (target_length + length_tolerance) || 
        length < (target_length - length_tolerance))
      continue;

    // check num of points
    if (lines_descriptors->at(idx).num_points < min_num_points)
      continue;

    // comfirm that pass
    std::cout << "FOUND Target at index: " << idx << std::endl;
    target_line_idx = idx;
    x_coor = mid_x;
    y_coor = mid_y;
    theta = lines_descriptors->at(idx).theta + PI / 2;

    // manage -ve senario
    if (theta < PI / 2)
      theta = PI + theta;

    std::cout << " - Found Target Pose!! : " << x_coor << " " 
              << y_coor << " " << theta << std::endl;
    break;
  }

  TargetPose[0] = x_coor;
  TargetPose[1] = y_coor;
  TargetPose[2] = theta;

  // For applyMovingAvgFiltering: buffer for target pose, for averaging
  targetPoseArray->push_back(TargetPose);
  // // after reachin avgin span, remove first ele of array before adding new ele
  if (targetPoseArray->size() > averaging_span)
    targetPoseArray->erase(targetPoseArray->begin());
}

// user get pose estimation
void ObjectPoseEstimate2D::getTargetPose(Eigen::Vector3f *target_pose)
{

  (*target_pose)[0] = TargetPose[0];
  (*target_pose)[1] = TargetPose[1];
  (*target_pose)[2] = TargetPose[2];
}

void ObjectPoseEstimate2D::getTargetPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  if (target_line_idx != -1)
  {
    *cloud = *(lines_cloud->at(target_line_idx));
  }
  else
  {
    cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  }
}

// TODO: get region of interest
void ObjectPoseEstimate2D::getROI(std::vector<Eigen::Vector3f> *roi_points)
{
  /// [x.min, x.max, y.min, y.max]
  // roi_range[0] = config["region_of_interest"]["x_min"].as<float>(); 
  // roi_range[1] = config["region_of_interest"]["x_max"].as<float>();;
  // roi_range[2] = config["region_of_interest"]["y_min"].as<float>();;
  // roi_range[3] = config["region_of_interest"]["y_max"].as<float>();;
  std::cout << "Getting region of interest" << std::endl;
}

// set input pointcloud
// this func will run all those func related to target pose estimation
void ObjectPoseEstimate2D::setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  input_cloud = cloud;
  std::cout << "[Obj Pose Estimation]::Num of Input Points: " << input_cloud->size() << "\n"
            << std::endl;
  // pcl processing
  objectClustering();
  lineFitting();
  findTargetPose();
}

// moving avg filtering
// This function is important to create a low and high pass filter 
// to smoothen the pose result while running in realtime inputs
void ObjectPoseEstimate2D::applyMovingAvgFiltering()
{

  int array_size = targetPoseArray->size();
  float jump_score = 0;

  // **Pose jump filtering*
  if (array_size >= 2)
  { // make sure there's prev pose to compare with current pose
    Eigen::Vector3f targetPoseDiff = 
      targetPoseArray->at(array_size - 1) - targetPoseArray->at(array_size - 2);
      // sumation of the square of x, y, yaw
    jump_score = targetPoseDiff.transpose() * targetPoseDiff; 
    std::cout << "## Jump score: " << jump_score << std::endl;
  }

  // check score, score > thresh means it's a jump
  if (jump_score > jump_score_thresh)
  {
    // check jump count on how many samples that have jumped
    if (jump_count < jump_count_allowance)
    {
      targetPoseArray->erase(targetPoseArray->end());
      jump_count++;
      std::cout << "## Increses jump count to " 
                << jump_count << ", with pose array size: " 
                << targetPoseArray->size() << std::endl;
    }
    else
    {
      jump_count = 0;
      targetPoseArray->erase(targetPoseArray->begin(), targetPoseArray->end() - 1);
      std::cout << "## Jump Reseted, with pose array size: " 
                << targetPoseArray->size() << std::endl;
    }
  }
  else
    jump_count = 0;

  // **Moving Averaging Filtering*
  Eigen::Vector3f target_pose_sums(0, 0, 0);
  array_size = targetPoseArray->size(); // refind the array size

  for (int i = 0; i < array_size; i++)
  {
    target_pose_sums += targetPoseArray->at(i);
  }

  // update output targetPose
  TargetPose[0] = target_pose_sums[0] / array_size;
  TargetPose[1] = target_pose_sums[1] / array_size;
  TargetPose[2] = target_pose_sums[2] / array_size;

  std::cout << "Applyfiltering with size: " << array_size << std::endl;
  std::cout << "- Ouput Pose, x, y, theta: " << TargetPose[0] 
            << " " << TargetPose[1] << " " << TargetPose[2] << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////// ------------------ Main Function ------------------- /////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // initialize PointClouds
  auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  auto target = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  Eigen::Vector3f target_pose;

  if (argc < 3 || pcl::console::find_switch(argc, argv, "-h"))
  {
    cout << " - Run this script to process .pcd file to ransac seg " << endl;
    cout << "Usage: ./random_sample_consensus -l -input <input_file>" << endl;
    return -1;
  }

  //get arg pcdinput
  if (pcl::console::find_argument(argc, argv, "-input") >= 0)
  {
    int input_idx = pcl::console::find_argument(argc, argv, "-input") + 1;
    std::string input_file = argv[input_idx];
    std::cout << "File Path: " << input_file << std::endl;
    pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, *cloud);
    std::cout << "Point cloud loaded, point size = " << cloud->points.size() << std::endl;
  }
  else
  {
    std::cout << "No Input PCD File, pls input via '-input' " << std::endl;
    exit(0);
  }

  // =============== Place class here =====================
  ObjectPoseEstimate2D agv_laser_scan("../config.yaml");
  agv_laser_scan.setInputCloud(cloud);
  // agv_laser_scan.applyMovingAvgFiltering(); // Use only on realtime multiple calls
  agv_laser_scan.getTargetPose(&target_pose);
  agv_laser_scan.getTargetPointCloud(target);

  std::cout << " Target Object Pose: " 
            << target_pose[0] << " "
            << target_pose[1] << " " 
            << target_pose[2] << std::endl;
  std::cout << " - target size: " << target->size() << "\n"
            << std::endl;

  // visualizer
  pcl::visualization::PCLVisualizer::Ptr viewer;
  viewer = agv_laser_scan.simpleVis();

  while (!viewer->wasStopped())
  {
    viewer->spinOnce(100);
  }
  return 0;
}
