#pragma once

#include <vector>
#include <unordered_map>
#include "../../extra/cloud/cloud.h"
#include <math.h>

class SampledData {
public:
  int count;
  PointXYZ point;

  SampledData() {
    count = 0;
    point = PointXYZ();
  }

  void update(const PointXYZ& p) {
    count += 1;
    point += p;
  }
};

class SampledListPoints {
public:
  int count;
  PointXYZ point;
  std::vector<PointXYZ> points;
  PointXYZ point_avg;

  SampledListPoints() {
    count = 0;
    point = PointXYZ();
    points = std::vector<PointXYZ>();
  }

  void update(const PointXYZ& p) {
    count += 1;
    points.push_back(p);
    point += p;
  }

  PointXYZ choose() {
    // int selected_index = rand() % count;
    // point = points[selected_index];
    // return point;

    // Find the closest point from the vector of points.
    point_avg = point * (1.0 / count);
    PointXYZ closestPoint = points[0];
    double minDistance = sqrt((closestPoint.x - point_avg.x)*(closestPoint.x - point_avg.x) + \
                              (closestPoint.y - point_avg.y)*(closestPoint.y - point_avg.y) + \
                              (closestPoint.z - point_avg.z)*(closestPoint.z - point_avg.z));
    for (auto& p : points) {
      double distance = sqrt((p.x - point_avg.x)*(p.x - point_avg.x) + \
                             (p.y - point_avg.y)*(p.y - point_avg.y) + \
                             (p.z - point_avg.z)*(p.z - point_avg.z));
      if (distance < minDistance) {
        minDistance = distance;
        closestPoint = p;
      }
    }
    return closestPoint;
  }
};

void single_grid_subsampling_cpu(
  std::vector<PointXYZ>& o_points,
  std::vector<PointXYZ>& s_points,
  float voxel_size
);

void grid_subsampling_cpu(
  std::vector<PointXYZ>& o_points,
  std::vector<PointXYZ>& s_points,
  std::vector<long>& o_lengths,
  std::vector<long>& s_lengths,
  float voxel_size
);

