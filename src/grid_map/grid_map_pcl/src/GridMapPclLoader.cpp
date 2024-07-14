/*
 * GridMapPclLoader.cpp
 *
 *  Created on: Aug 26, 2019
 *      Author: Edo Jelavic
 *      Institute: ETH Zurich, Robotic Systems Lab
 */

#include <chrono>

#ifdef GRID_MAP_PCL_OPENMP_FOUND
#include <omp.h>
#endif

#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h> 
#include <pcl/conversions.h>   
#include <pcl_ros/transforms.h>
#include <ros/console.h>

#include <sensor_msgs/PointCloud2.h>

#include <grid_map_core/GridMapMath.hpp>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

#include "grid_map_pcl/GridMapPclLoader.hpp"
#include "grid_map_pcl/helpers.hpp"



namespace grid_map {

const grid_map::GridMap& GridMapPclLoader::getGridMap() const {
  return workingGridMap_;
}

void GridMapPclLoader::loadCloudFromPcdFile(const std::string& filename) {
  Pointcloud::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZ>);
  inputCloud = grid_map_pcl::loadPointcloudFromPcd(filename);
  setInputCloud(inputCloud);
}



void GridMapPclLoader::setInputCloud(Pointcloud::ConstPtr inputCloud) {
  setRawInputCloud(inputCloud);
  setWorkingCloud(inputCloud);
}

void GridMapPclLoader::setRawInputCloud(Pointcloud::ConstPtr rawInputCloud) {
  rawInputCloud_.reset();
  Pointcloud::Ptr temp(new Pointcloud());
  pcl::copyPointCloud(*rawInputCloud, *temp);
  rawInputCloud_ = temp;
}

void GridMapPclLoader::setWorkingCloud(Pointcloud::ConstPtr workingCloud) {
  workingCloud_.reset();
  Pointcloud::Ptr temp(new Pointcloud());
  pcl::copyPointCloud(*workingCloud, *temp);
  workingCloud_ = temp;
}

void GridMapPclLoader::preProcessInputCloud() {
  // Preprocess: Remove outliers, downsample cloud, transform cloud
  ROS_INFO_STREAM("Preprocessing of the pointcloud started");

  if (params_.get().outlierRemoval_.isRemoveOutliers_) {
    auto filteredCloud = pointcloudProcessor_.removeOutliersFromInputCloud(workingCloud_);
    setWorkingCloud(filteredCloud);
  }

  if (params_.get().downsampling_.isDownsampleCloud_) {
    auto downsampledCloud = pointcloudProcessor_.downsampleInputCloud(workingCloud_);
    setWorkingCloud(downsampledCloud);
  }

  auto transformedCloud1 = pointcloudProcessor_.applyRigidBodyTransformation(workingCloud_);
  setWorkingCloud(transformedCloud1);
  auto filtedCloud = pointcloudProcessor_.filtFromInputCloud(workingCloud_);
  setWorkingCloud(filtedCloud);
  auto transformedCloud2 = pointcloudProcessor_.applyReverseRigidBodyTransformation(workingCloud_);
  setWorkingCloud(transformedCloud2);

  ROS_INFO_STREAM("Preprocessing and filtering finished");
}

void GridMapPclLoader::initializeGridMapGeometryFromInputCloud() {
  workingGridMap_.clearAll();
  const double resolution = params_.get().gridMap_.resolution_;
  if (resolution < 1e-4) {
    throw std::runtime_error("Desired grid map resolution is zero");
  }

  // find point cloud dimensions
  // min and max coordinate in x,y and z direction
  pcl::PointXYZ minBound;
  pcl::PointXYZ maxBound;
  pcl::getMinMax3D(*workingCloud_, minBound, maxBound);

  // from min and max points we can compute the length
  grid_map::Length length = grid_map::Length(maxBound.x - minBound.x, maxBound.y - minBound.y);

  // we put the center of the grid map to be in the middle of the point cloud
  grid_map::Position position = grid_map::Position((maxBound.x + minBound.x) / 2.0, (maxBound.y + minBound.y) / 2.0);
  workingGridMap_.setGeometry(length, resolution, position);

  ROS_INFO_STREAM("Grid map dimensions: " << workingGridMap_.getLength()(0) << " x " << workingGridMap_.getLength()(1));
  ROS_INFO_STREAM("Grid map resolution: " << workingGridMap_.getResolution());
  ROS_INFO_STREAM("Grid map num cells: " << workingGridMap_.getSize()(0) << " x " << workingGridMap_.getSize()(1));
  ROS_INFO_STREAM("Initialized map geometry");
}

void GridMapPclLoader::addLayerFromInputCloud(const std::string& layer) {
  ROS_INFO_STREAM("Started adding layer: " << layer);
  // Preprocess: allocate memory in the internal data structure
  preprocessGridMapCells();
  ROS_INFO("Finished preprocessing");
  workingGridMap_.add(layer);
  grid_map::GridMap tempMap;
  tempMap = workingGridMap_;
  grid_map::Matrix& tempMapData = tempMap.get(layer);
  grid_map::Matrix& gridMapData = workingGridMap_.get(layer);
  unsigned int linearGridMapSize = workingGridMap_.getSize().prod();

  // Iterate through grid map and calculate the corresponding height based on the point cloud
#ifndef GRID_MAP_PCL_OPENMP_FOUND
  ROS_WARN_STREAM("OpemMP not found, defaulting to single threaded implementation");
#else
  omp_set_num_threads(params_.get().numThreads_);
#pragma omp parallel for schedule(dynamic, 10)
#endif
  for (unsigned int linearIndex = 0; linearIndex < linearGridMapSize; ++linearIndex) {
    processGridMapCell(linearIndex, &gridMapData);
  }
  // liji: linear interpolation
  for (unsigned int linearIndex = 0; linearIndex < linearGridMapSize; ++linearIndex) {
    
    interpolationGridMapCell(linearIndex, &tempMapData, &gridMapData);
  }

  

  ROS_INFO_STREAM("Finished adding layer: " << layer);
}

void GridMapPclLoader::interpolationGridMapCell(const unsigned int linearGridMapIndex, grid_map::Matrix* tempMapData, grid_map::Matrix* gridMapData) {
  // Get grid map index from linear index and check if enough points lie within the cell
  const grid_map::Index index(grid_map::getIndexFromLinearIndex(linearGridMapIndex, workingGridMap_.getSize()));
  int rows = workingGridMap_.getSize()(0);
  int cols = workingGridMap_.getSize()(1);
  // std::cout << "rows left" << rows << std::endl;
  // std::cout << "cols front"  << cols << std::endl;
  // if (index(0) > 401){
  //   std::cout << index(0) << std::endl;
  // }
  const double resolution = params_.get().gridMap_.resolution_;
  // if (index(0) < (rows/2 + 20/resolution) && index(0) > (rows/2 - 20/resolution) && index(1) > (70/resolution - 30/resolution)){
  if(std::isnan((*tempMapData)(index(0), index(1)))){
  std::vector<std::pair<int, int>> nearestPairs = findNearestValidValues(linearGridMapIndex, tempMapData);

  int nearestRow = nearestPairs[0].first;
  int nearestCol = nearestPairs[0].second;
  int secondNearestRow = nearestPairs[1].first;
  int secondNearestCol = nearestPairs[1].second;
  int thirdNearestRow = nearestPairs[2].first;
  int thirdNearestCol = nearestPairs[2].second;
  int forthNearestRow = nearestPairs[3].first;
  int forthNearestCol = nearestPairs[3].second;

  double nearestValue = (*tempMapData)(nearestRow, nearestCol);
  double secondNearestValue = (*tempMapData)(secondNearestRow, secondNearestCol);
  double thirdNearestValue = (*tempMapData)(thirdNearestRow, thirdNearestCol);
  double forthNearestValue = (*tempMapData)(forthNearestRow, forthNearestCol);


  if (nearestRow != -1 && nearestCol != -1) {
    // (*gridMapData)(index(0), index(1)) = bilinearInterpolation(nearestRow, forthNearestCol, secondNearestRow, thirdNearestCol,
    //                                                           nearestValue, secondNearestValue, thirdNearestValue, forthNearestValue,
    //                                                           index(0), index(1))  
      (*gridMapData)(index(0), index(1)) = ( nearestValue + secondNearestValue + thirdNearestValue + forthNearestValue ) / 4.0;
    }
  // }
  }
  
}

double bilinearInterpolation(double x1, double y1, double x2, double y2,
                             double q11, double q12, double q21, double q22,
                             double x, double y) {

    double xDist = x2 - x1;
    double yDist = y2 - y1;
    double dx = x - x1;
    double dy = y - y1;

    double f11 = q11 * (xDist - dx) * (yDist - dy);
    double f12 = q12 * (xDist - dx) * dy;
    double f21 = q21 * dx * (yDist - dy);
    double f22 = q22 * dx * dy;

    return (f11 + f12 + f21 + f22) / (xDist * yDist);
}

std::vector<std::pair<int, int>> GridMapPclLoader::findNearestValidValues(const unsigned int linearGridMapIndex, grid_map::Matrix* gridMapData) {
    const grid_map::Index index(grid_map::getIndexFromLinearIndex(linearGridMapIndex, workingGridMap_.getSize()));
    int row = index(0);
    int col = index(1);
    int rows = workingGridMap_.getSize()(0);
    int cols = workingGridMap_.getSize()(1);
    double minDistance1 = std::numeric_limits<double>::max();
    double minDistance2 = std::numeric_limits<double>::max();
    double minDistance3 = std::numeric_limits<double>::max();
    double minDistance4 = std::numeric_limits<double>::max();
    std::pair<int, int> nearestPair = std::make_pair(-1, -1);
    std::pair<int, int> secondNearestPair = std::make_pair(-1, -1);
    std::pair<int, int> thirdNearestPair = std::make_pair(-1, -1);
    std::pair<int, int> fourthNearestPair = std::make_pair(-1, -1);
    for (int i = std::max((row - 5), 0); i < std::min((row + 5), rows); i++) {
      for (int j = std::max((col - 5), 0); j < std::min((col + 5), cols); j++) {
      if(!std::isnan((*gridMapData)(i, j))) {
          double distance = calculateDistance(index(0), index(1), i, j);
          if (distance < minDistance1) {
            minDistance4 = minDistance3;
            fourthNearestPair = thirdNearestPair;
            minDistance3 = minDistance2;
            thirdNearestPair = secondNearestPair;
            minDistance2 = minDistance1;
            secondNearestPair = nearestPair;
            minDistance1 = distance;
            nearestPair = std::make_pair(i, j);
          } else if (distance < minDistance2) {
            minDistance4 = minDistance3;
            fourthNearestPair = thirdNearestPair;
            minDistance3 = minDistance2;
            thirdNearestPair = secondNearestPair;
            minDistance2 = distance;
            secondNearestPair = std::make_pair(i, j);
          } else if (distance < minDistance3) {
            minDistance4 = minDistance3;
            fourthNearestPair = thirdNearestPair;
            minDistance3 = distance;
            thirdNearestPair = std::make_pair(i, j);
          } else if (distance < minDistance4) {
            minDistance4 = distance;
            fourthNearestPair = std::make_pair(i, j);
          }
        }
      }
    }
    // 返回最近邻非 NaN 值的索引
    std::vector<std::pair<int, int>> nearestPairs = {nearestPair, secondNearestPair, thirdNearestPair, fourthNearestPair};
    return nearestPairs;
}


double GridMapPclLoader::calculateDistance(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

std::pair<int, int> GridMapPclLoader::findNearestValidValue(const unsigned int linearGridMapIndex, grid_map::Matrix* gridMapData) {
    const grid_map::Index index(grid_map::getIndexFromLinearIndex(linearGridMapIndex, workingGridMap_.getSize()));
    int row = index(0);
    int col = index(1);
    int rows = workingGridMap_.getSize()(0);
    int cols = workingGridMap_.getSize()(1);
    double minDistance = std::numeric_limits<double>::max();
    int nearestRow = -1;
    int nearestCol = -1;
    for (int i = std::max((row - 5), 0); i < std::min((row + 5), rows); i++) {
        for (int j = std::max((col - 5), 0); j < std::min((col + 5), cols); j++) {
        if(!std::isnan((*gridMapData)(i, j))) {
            double distance = calculateDistance(index(0), index(1), i, j);
            if (distance < minDistance) {
                minDistance = distance;
                nearestRow = i;
                nearestCol = j;
            } 
        }
        }
    }
    // 返回最近邻非 NaN 值的索引
    return std::make_pair(nearestRow, nearestCol);
}

// 填充二维矩阵中的 NaN 值
void GridMapPclLoader::fillNaNGridMapCell(const unsigned int linearGridMapIndex, grid_map::Matrix* gridMapData) {
    const grid_map::Index index(grid_map::getIndexFromLinearIndex(linearGridMapIndex, workingGridMap_.getSize()));
    int rows = workingGridMap_.getSize()(0);
    int cols = workingGridMap_.getSize()(1);
    if(std::isnan((*gridMapData)(index(0), index(1)))){
      std::pair<int, int> nearestValidIdx = findNearestValidValue(linearGridMapIndex, gridMapData);
      int nearestRow = nearestValidIdx.first;
      int nearestCol = nearestValidIdx.second;
      if (nearestRow != -1 && nearestCol != -1) {
        (*gridMapData)(index(0), index(1)) = (*gridMapData)(nearestRow, nearestCol);      
      
        }
      }
 }


double GridMapPclLoader::linearInterpolation(double x1, double y1, double x2, double y2, double x) {
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
}


void GridMapPclLoader::processGridMapCell(const unsigned int linearGridMapIndex, grid_map::Matrix* gridMapData) {
  // Get grid map index from linear index and check if enough points lie within the cell
  const grid_map::Index index(grid_map::getIndexFromLinearIndex(linearGridMapIndex, workingGridMap_.getSize()));

  Pointcloud::Ptr pointsInsideCellBorder(new Pointcloud());
  pointsInsideCellBorder = getPointcloudInsideGridMapCellBorder(index);
  const bool isTooFewPointsInCell = pointsInsideCellBorder->size() < params_.get().gridMap_.minCloudPointsPerCell_;
  if (isTooFewPointsInCell) {
    ROS_WARN_STREAM_THROTTLE(10.0, "Less than " << params_.get().gridMap_.minCloudPointsPerCell_ << " points in a cell. Skipping.");
    return;
  }
  if (pointsInsideCellBorder->size() > params_.get().gridMap_.maxCloudPointsPerCell_) {
    ROS_WARN_STREAM_THROTTLE(10.0, "More than " << params_.get().gridMap_.maxCloudPointsPerCell_ << " points in a cell. Skipping.");
    return;
  }
  
  if(params_.get().clusterExtraction_.useCluster_){ // liji
    // ROS_INFO("Using Cluster!");
    auto& clusterHeights = clusterHeightsWithingGridMapCell_[index(0)][index(1)];
    calculateElevationFromPointsInsideGridMapCell(pointsInsideCellBorder, clusterHeights); 
    if (clusterHeights.empty()) {
    (*gridMapData)(index(0), index(1)) = std::nan("1");
    } else {
    (*gridMapData)(index(0), index(1)) = params_.get().clusterExtraction_.useMaxHeightAsCellElevation_
                                             ? *(std::max_element(clusterHeights.begin(), clusterHeights.end()))
                                             : *(std::min_element(clusterHeights.begin(), clusterHeights.end()));
    }
  }
  else{
    float heights;
    heights = calculateElevationFromPointsInsideGridMapCellNoCluster(pointsInsideCellBorder); // liji
    (*gridMapData)(index(0), index(1)) = heights;
  }
  
  
  
}
float GridMapPclLoader::calculateElevationFromPointsInsideGridMapCellNoCluster(Pointcloud::ConstPtr cloud) const {
  // return grid_map_pcl::calculateMiniumMeanOfPointPositions(cloud).z();
  return grid_map_pcl::calculateMeanOfPointPositions(cloud).z();

}

void GridMapPclLoader::calculateElevationFromPointsInsideGridMapCell(Pointcloud::ConstPtr cloud, std::vector<float>& heights) const {
  heights.clear();
  // Extract point cloud cluster from point cloud and return if none is found.
  std::vector<Pointcloud::Ptr> clusterClouds = pointcloudProcessor_.extractClusterCloudsFromPointcloud(cloud);
  const bool isNoClustersFound = clusterClouds.empty();
  if (isNoClustersFound) {
    ROS_WARN_STREAM_THROTTLE(10.0, "No clusters found in the grid map cell");
    return;
  }

  // Extract mean z value of cluster vector and return smallest height value
  heights.reserve(clusterClouds.size());

  std::transform(clusterClouds.begin(), clusterClouds.end(), std::back_inserter(heights),
                 [this](Pointcloud::ConstPtr cloud) -> double { return grid_map_pcl::calculateMeanOfPointPositions(cloud).z(); });
}

GridMapPclLoader::Pointcloud::Ptr GridMapPclLoader::getPointcloudInsideGridMapCellBorder(const grid_map::Index& index) const {
  return pointcloudWithinGridMapCell_[index.x()][index.y()];
}

void GridMapPclLoader::loadParameters(const std::string& filename) {
  params_.loadParameters(filename);
  pointcloudProcessor_.loadParameters(filename);
}

void GridMapPclLoader::setParameters(grid_map_pcl::PclLoaderParameters::Parameters parameters) {
  params_.parameters_ = std::move(parameters);
}

void GridMapPclLoader::savePointCloudAsPcdFile(const std::string& filename) const {
  pointcloudProcessor_.savePointCloudAsPcdFile(filename, *workingCloud_);
}

void GridMapPclLoader::preprocessGridMapCells() {
  allocateSpaceForCloudsInsideCells();
  dispatchWorkingCloudToGridMapCells();
}

void GridMapPclLoader::allocateSpaceForCloudsInsideCells() {
  const unsigned int dimX = workingGridMap_.getSize().x() + 1;
  const unsigned int dimY = workingGridMap_.getSize().y() + 1;

  // resize vectors
  pointcloudWithinGridMapCell_.resize(dimX);
  clusterHeightsWithingGridMapCell_.resize(dimX);

  // allocate pointClouds
  for (unsigned int i = 0; i < dimX; ++i) {
    pointcloudWithinGridMapCell_[i].resize(dimY);
    clusterHeightsWithingGridMapCell_[i].resize(dimY);
    for (unsigned int j = 0; j < dimY; ++j) {
      pointcloudWithinGridMapCell_[i][j].reset(new Pointcloud());
      clusterHeightsWithingGridMapCell_[i][j].clear();
    }
  }
}

void GridMapPclLoader::dispatchWorkingCloudToGridMapCells() {
  // For each point in input point cloud, find which grid map
  // cell does it belong to. Then copy that point in the
  // right cell in the matrix of point clouds data structure.
  // This allows for faster access in the clustering stage.
  for (unsigned int i = 0; i < workingCloud_->points.size(); ++i) {
    const Point& point = workingCloud_->points[i];
    const double x = point.x;
    const double y = point.y;
    grid_map::Index index;
    workingGridMap_.getIndex(grid_map::Position(x, y), index);
    pointcloudWithinGridMapCell_[index.x()][index.y()]->push_back(point);
  }
}

}
  // namespace grid_map
