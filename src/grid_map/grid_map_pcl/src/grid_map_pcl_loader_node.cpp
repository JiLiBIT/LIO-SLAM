/*
 * grid_map_pcl_loader_node.cpp
 *
 *  Created on: Aug 26, 2019
 *      Author: Edo Jelavic
 *      Institute: ETH Zurich, Robotic Systems Lab
 */

#include <ros/ros.h>
#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h> 
#include <pcl/conversions.h>   
#include <pcl_ros/transforms.h>
#include <ros/console.h>


#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <sensor_driver_msgs/mapCloudwithGps.h>
#include <sensor_driver_msgs/GpswithHeading.h>

#include "grid_map_pcl/GridMapPclLoader.hpp"
#include "grid_map_pcl/helpers.hpp"

namespace gm = ::grid_map::grid_map_pcl;
using Point = ::pcl::PointXYZ;
using Pointcloud = ::pcl::PointCloud<Point>;

class gridmapNode {
public:
  ros::NodeHandle nh; 
  ros::Subscriber pointCloudSub;
  ros::Subscriber poseSub;
  ros::Publisher gridMapPub;
  ros::Publisher pubTransformed;

  gridmapNode()
  {
    pointCloudSub = nh.subscribe<sensor_msgs::PointCloud2>("liorf/mapping/map_4planning", 1, &gridmapNode::cloudMapInfoHandler, this, ros::TransportHints().tcpNoDelay());
    poseSub = nh.subscribe<sensor_driver_msgs::GpswithHeading>("liorf/gpsdata", 1, &gridmapNode::poseInfoHandler, this, ros::TransportHints().tcpNoDelay());
    gridMapPub = nh.advertise<grid_map_msgs::GridMap>("/height_map", 1, true);
    pubTransformed           = nh.advertise<sensor_msgs::PointCloud2> ("/transformed_pointcloud", 1);
  }

  void cloudMapInfoHandler(const sensor_msgs::PointCloud2ConstPtr& cloudMsg)
  {
    grid_map::GridMapPclLoader gridMapPclLoader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloudMsg, *inputCloud);
    gridMapPclLoader.loadParameters(gm::getParameterPath(nh));
    gridMapPclLoader.setInputCloud(inputCloud);

    
    gm::processPointcloud(&gridMapPclLoader, nh);

    

    sensor_msgs::PointCloud2 Cloudmsg;
    pcl::toROSMsg(*gm::filtedCloudPub, Cloudmsg);
    Cloudmsg.header.stamp = ros::Time::now();
    Cloudmsg.header.frame_id = "map";
    pubTransformed.publish(Cloudmsg);

    grid_map::GridMap gridMap;
    gridMap = gridMapPclLoader.getGridMap();
    gridMap.setFrameId(gm::getMapFrame(nh));
    
    grid_map_msgs::GridMap msg;
    grid_map::GridMapRosConverter::toMessage(gridMap, msg);

    gridMapPub.publish(msg);
  }

  void poseInfoHandler(const sensor_driver_msgs::GpswithHeadingConstPtr& poseMsg)
  {
    std::cout<< "222" << std::endl;
    float rollAngle = poseMsg->roll;
    float pitchAngle = poseMsg->pitch;
    float yawAngle = poseMsg->heading;
    gm::thisPoseYaw = yawAngle * std::acos(-1.0) / 180.0;
    gm::thisPosePitch = pitchAngle * std::acos(-1.0) / 180.0;
    gm::thisPoseRoll = rollAngle * std::acos(-1.0) / 180.0;

  }
};


int main(int argc, char** argv) {
  ros::init(argc, argv, "grid_map_pcl_loader_node");
  gridmapNode GN;

  
  
  

  // pointCloudSub =nh.subscribe<sensor_driver_msgs::mapCloudwithGps>("/liorf/mapping/cloud_gps", 1, boost::bind(&cloudMapInfoHandler, _1, nh));
  
  

  // liji ros pcl2
  
  ros::spin();

  return 0;

  // // run
  // ros::spin();
  // return EXIT_SUCCESS;
}
