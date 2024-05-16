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
  PointXYZ normal;
  std::vector<PointXYZ> normals;
  PointXYZ point_avg;

  SampledListPoints() {
    count = 0;
    point = PointXYZ();
    points = std::vector<PointXYZ>();
    normal = PointXYZ();
    normals = std::vector<PointXYZ>();
  }

  void update(const PointXYZ& p, const PointXYZ& n) {
    count += 1;
    points.push_back(p);
    point += p;
    normals.push_back(n);
    normal += n;
  }

  void choose(PointXYZ& closestPoint, PointXYZ& closestNormal) {
    // int selected_index = rand() % count;
    // point = points[selected_index];
    // return point;

    // Find the closest point from the vector of points.
    point_avg = point * (1.0 / count);
    closestPoint = points[0];
    closestNormal = normals[0];
    double minDistance = sqrt((closestPoint.x - point_avg.x)*(closestPoint.x - point_avg.x) + \
                              (closestPoint.y - point_avg.y)*(closestPoint.y - point_avg.y) + \
                              (closestPoint.z - point_avg.z)*(closestPoint.z - point_avg.z));
    // for (auto& p : points) {
    for (int j = 0; j < points.size(); j++) {
      PointXYZ p = points[j];
      double distance = sqrt((p.x - point_avg.x)*(p.x - point_avg.x) + \
                             (p.y - point_avg.y)*(p.y - point_avg.y) + \
                             (p.z - point_avg.z)*(p.z - point_avg.z));
      if (distance < minDistance) {
        minDistance = distance;
        closestPoint = p;
        closestNormal = normals[j];
      }
    }
  }
};

void single_grid_subsampling_cpu(
  std::vector<PointXYZ>& o_points,
  std::vector<PointXYZ>& s_points,
  std::vector<PointXYZ>& o_normals,
  std::vector<PointXYZ>& s_normals,
  float voxel_size
);

void grid_subsampling_cpu(
  std::vector<PointXYZ>& o_points,
  std::vector<PointXYZ>& s_points,
  std::vector<long>& o_lengths,
  std::vector<long>& s_lengths,
  std::vector<PointXYZ>& o_normals,
  std::vector<PointXYZ>& s_normals,
  float voxel_size
);

