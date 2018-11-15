#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>


int pointSize = 4; //default point size for visualization

// Load file | Works with PCD and PLY files
pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud2 (new pcl::PointCloud<pcl::PointXYZ> ());
pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud3 (new pcl::PointCloud<pcl::PointXYZ> ());


// This function displays the help
void showHelp(char * program_name)
{
  std::cout << std::endl;
  std::cout << "Usage: " << program_name << " cloud_filename.[pcd|ply]" << std::endl;
  std::cout << "-s:  for size changes." << std::endl;
  std::cout << "-h:  Show this help." << std::endl;
  std::cout << " * Could visualize up to 3 point clouds path" << std::endl;
}


//visuallization
void visualization(int filesize)
{
  pcl::visualization::PCLVisualizer viewer ("PC visualizer");

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (source_cloud, 250, 23, 23);
  viewer.addPointCloud (source_cloud, source_cloud_color_handler, "original_cloud");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, "original_cloud");

  // pointcloud 2
  if (filesize >= 2){
    std::cout << "[ Point Cloud 2 ] Visualizing" << std::endl;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler2 (source_cloud2, 23, 255, 255);
    viewer.addPointCloud (source_cloud2, source_cloud_color_handler2, "original_cloud2");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud2");
  }

  // pointcloud 3
  if (filesize >= 3){
    std::cout << "[ Point Cloud 3 ] Visualizing" << std::endl;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler3 (source_cloud3, 23, 255, 23);
    viewer.addPointCloud (source_cloud3, source_cloud_color_handler3, "original_cloud3");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "original_cloud3");
  }

  viewer.addCoordinateSystem (1.0, "cloud", 0);
  viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey

  while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
    viewer.spinOnce ();
  }
}


// This is the main function
int main (int argc, char** argv)
{
  // Show help
  if (pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")) {
    showHelp (argv[0]);
    return 0;
  }

  //get arg point size
  if (pcl::console::find_switch (argc, argv, "-s")){
    int input_idx = pcl::console::find_argument (argc, argv, "-s") + 1;
    pointSize = std::atoi(argv[input_idx]);
    std::cout << "Input Int arg for '-s' is " << pointSize << std::endl;
  } 

  // Fetch point cloud filename in arguments | Works with PCD and PLY files
  std::vector<int> filenames;
  bool file_is_pcd = true;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  std::cout << "<init> input pcd route num is " << filenames.size() << std::endl;


  //first point cloud
  if (file_is_pcd) {
    std::cout << "loading point cloud: " << argv[filenames[0]] << std::endl;  
    if (pcl::io::loadPCDFile (argv[filenames[0]], *source_cloud) < 0)  {
      std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
      showHelp (argv[0]);
      return -1;
    }
  } else {
    if (pcl::io::loadPLYFile (argv[filenames[0]], *source_cloud) < 0)  {
      std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
      showHelp (argv[0]);
      return -1;
    }
  }

  // second point cloud 
  if (filenames.size() >= 2){
    std::cout << "loading 2nd point cloud: " << argv[filenames[1]] << std::endl;
    if (file_is_pcd) {
      if (pcl::io::loadPCDFile (argv[filenames[1]], *source_cloud2) < 0)  {
        std::cout << "Error loading point cloud " << argv[filenames[1]] << std::endl << std::endl;
        showHelp (argv[1]);
        return -1;
      }
    } else {
      if (pcl::io::loadPLYFile (argv[filenames[1]], *source_cloud2) < 0)  {
        std::cout << "Error loading point cloud " << argv[filenames[1]] << std::endl << std::endl;
        showHelp (argv[1]);
        return -1;
      }
    }
  }

  // third point cloud 
  if (filenames.size() >= 3){
    std::cout << "loading 3nd point cloud: " << argv[filenames[2]] << std::endl;
    if (file_is_pcd) {
      if (pcl::io::loadPCDFile (argv[filenames[2]], *source_cloud3) < 0)  {
        std::cout << "Error loading point cloud " << argv[filenames[2]] << std::endl << std::endl;
        showHelp (argv[2]);
        return -1;
      }
    } else {
      if (pcl::io::loadPLYFile (argv[filenames[2]], *source_cloud3) < 0)  {
        std::cout << "Error loading point cloud " << argv[filenames[2]] << std::endl << std::endl;
        showHelp (argv[2]);
        return -1;
      }
    }
  }

  std::cout << "[COLOR]:: Red: 1st; Blue: 2nd; Green: 3rd \n" << std::endl;

  visualization(filenames.size());
  return 0;
}
