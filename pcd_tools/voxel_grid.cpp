#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

// This function displays the help
void showHelp(char * program_name)
{
  std::cout << std::endl;
  std::cout << "Usage: " << program_name << " cloud_filename.[pcd]" << std::endl;
  std::cout << "-s:  for leafsize changes." << std::endl;
  std::cout << "-h:  Show this help." << std::endl;
}


int
main (int argc, char** argv)
{

  double leafsize = 0.05;

    // Show help
  if (pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")) {
    showHelp (argv[0]);
    return 0;
  }

  //get arg point size
  int pointSize = 5; //default
  if (pcl::console::find_switch (argc, argv, "-s")){
    int input_idx = pcl::console::find_argument (argc, argv, "-s") + 1;
    pointSize = std::atoi(argv[input_idx]);
    std::cout << "Input Int arg for '-s' is " << pointSize << std::endl;
  } 

  //get arg point size
  if (pcl::console::find_switch (argc, argv, "-s")){
      int input_idx = pcl::console::find_argument (argc, argv, "-s") + 1;
      std::stringstream ss( argv[input_idx] );
      if ( !(ss >> leafsize))
      std::cout << "Invalid double for -s...\n";
  } 
  std::cout << "leafsize is " << leafsize << std::endl;
  

  // Fetch point cloud filename in arguments | Works with PCD and PLY files
  std::vector<int> filenames;
  bool file_is_pcd = true;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  std::cout << "<init> input pcd route num is " << filenames.size() << std::endl;

  if (filenames.size() == 0) {
    showHelp (argv[0]);
    return 0;
  }

  pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
  pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());


  if (pcl::io::loadPCDFile (argv[filenames[0]], *cloud) < 0)  {
    std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
    showHelp (argv[0]);
    return -1;
  }

  // // ----- Start Program ---------


  // Fill in the cloud data
  pcl::PCDReader reader;
  
  std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height 
       << " data points (" << pcl::getFieldsList (*cloud) << ").\n";

  // Create the filtering object
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (leafsize, leafsize, leafsize);
  sor.filter (*cloud_filtered);

  std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height 
       << " data points (" << pcl::getFieldsList (*cloud_filtered) << ").\n";

  pcl::PCDWriter writer;
  writer.write ("downsampled.pcd", *cloud_filtered, 
         Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);

  return (0);
}
