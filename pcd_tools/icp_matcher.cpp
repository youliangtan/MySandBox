/*

  Created By Youliang Nov 2018
  Testing of pcl matching
  change parameter in icpMatching() function

*/

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>



void showHelp(char * program_name)
{
  std::cout << std::endl;
  std::cout << "Usage: " << program_name << " [cloud1.pcd] [cloud2.pcd]" << std::endl;
  std::cout << "\t\t first pointcloud being transformed to 2nd pointcloud  " << std::endl;

}


// uses and modified from odom based 3D SLAM
Eigen::VectorXf icpMatching(const pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud){
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

  // https://stackoverflow.com/questions/37853423/point-cloud-registration-using-pcl-iterative-closest-point
  icp.setInputSource(current_cloud);
  icp.setInputTarget(prev_cloud);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon (1e-9);
  icp.setRANSACOutlierRejectionThreshold (0.005);
  icp.setMaxCorrespondenceDistance (5);


  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);

  std::cout << "-- ICP has converged:" << icp.hasConverged() << "   score: " <<  icp.getFitnessScore() << std::endl;

  Eigen::Matrix4f tf_matrix = icp.getFinalTransformation(); // 4x4 matrix
  Eigen::Matrix3f rot_matrix = tf_matrix.block<3,3>(0,0);   // 3x3 Rot Matrix
  Eigen::Vector3f rpy = rot_matrix.eulerAngles(0, 1, 2);    // 1x3 rpy rotation
  std::cout << "-- ICP 4x4 Matrix: \n" << tf_matrix << std::endl;

  Eigen::Vector3f xyz; 
  xyz << tf_matrix(0,3), tf_matrix(1,3), tf_matrix(2,3);    // 1x3 xyz trans
  Eigen::VectorXf vec_xyzrpy(6);

  vec_xyzrpy << xyz, rpy;  
  std::cout << "XYZRPY : \n" << vec_xyzrpy << std::endl;

  return vec_xyzrpy;
}


int main (int argc, char** argv)
{
    // Show help
  if (pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")) {
    showHelp (argv[0]);
    return 0;
  }

  // Fetch point cloud filename in arguments | Works with PCD and PLY files
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  std::cout << "<init> input pcd route num is " << filenames.size() << std::endl;

  // Load file | Works with PCD and PLY files
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud2 (new pcl::PointCloud<pcl::PointXYZ> ());

  std::cout << "[READ] loading point cloud: " << argv[filenames[0]] << std::endl;  
  if (pcl::io::loadPCDFile (argv[filenames[0]], *source_cloud) < 0)  {
    std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
    showHelp (argv[0]);
    return -1;
  }

  // second point cloud 
  if (filenames.size() == 2){
    std::cout << "[READ] loading 2nd point cloud: " << argv[filenames[1]] << std::endl;
    if (pcl::io::loadPCDFile (argv[filenames[1]], *source_cloud2) < 0)  {
      std::cout << "Error loading point cloud " << argv[filenames[1]] << std::endl << std::endl;
      showHelp (argv[1]);
      return -1;
    }
  }


  // ==================   ICP Transformation  =====================

  // match current transformed cloud with prev cloud
  Eigen::VectorXf icp_tf(6);
  icp_tf = icpMatching(source_cloud, source_cloud2);

  Eigen::Affine3f transform = Eigen::Affine3f::Identity(); 

  // translation
  transform.translation() << icp_tf[0], icp_tf[1], icp_tf[2];

  // The same rotation matrix as before; theta radians around Z axis
  transform.rotate (Eigen::AngleAxisf (icp_tf[3], Eigen::Vector3f::UnitX()));
  transform.rotate (Eigen::AngleAxisf (icp_tf[4], Eigen::Vector3f::UnitY()));
  transform.rotate (Eigen::AngleAxisf (icp_tf[5], Eigen::Vector3f::UnitZ()));

  // // Executing the transformation
  pcl::PointCloud<pcl::PointXYZ>::Ptr icp_transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::transformPointCloud (*source_cloud, *icp_transformed_cloud, transform);

  // save pcd file
  pcl::io::savePCDFileASCII ("after_icp.pcd", *icp_transformed_cloud);
  std::cout<<" SUCCESS: Output .pcd file is saved!! "<<std::endl;

  // ==================== end ICP Transformation 2 =====================

  return (0);
}
