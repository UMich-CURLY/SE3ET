#include "grid_subsampling_cpu.h"

void single_grid_subsampling_cpu(
  std::vector<PointXYZ>& points,
  std::vector<PointXYZ>& s_points,
  std::vector<PointXYZ>& normals,
  std::vector<PointXYZ>& s_normals,
  float voxel_size
) {
//  float sub_scale = 1. / voxel_size;
  PointXYZ minCorner = min_point(points);
  PointXYZ maxCorner = max_point(points);
  PointXYZ originCorner = floor(minCorner * (1. / voxel_size)) * voxel_size;

  std::size_t sampleNX = static_cast<std::size_t>(
//    floor((maxCorner.x - originCorner.x) * sub_scale) + 1
    floor((maxCorner.x - originCorner.x) / voxel_size) + 1
  );
  std::size_t sampleNY = static_cast<std::size_t>(
//    floor((maxCorner.y - originCorner.y) * sub_scale) + 1
    floor((maxCorner.y - originCorner.y) / voxel_size) + 1
  );

  std::size_t iX = 0;
  std::size_t iY = 0;
  std::size_t iZ = 0;
  std::size_t mapIdx = 0;
  std::unordered_map<std::size_t, SampledData> data;
  std::unordered_map<std::size_t, SampledListPoints> data_choose;


  // for (auto& p : points) {
  for (int i = 0; i < points.size(); i++) {
    PointXYZ p = points[i];
    PointXYZ n = normals[i];
//    iX = static_cast<std::size_t>(floor((p.x - originCorner.x) * sub_scale));
//    iY = static_cast<std::size_t>(floor((p.y - originCorner.y) * sub_scale));
//    iZ = static_cast<std::size_t>(floor((p.z - originCorner.z) * sub_scale));
    iX = static_cast<std::size_t>(floor((p.x - originCorner.x) / voxel_size));
    iY = static_cast<std::size_t>(floor((p.y - originCorner.y) / voxel_size));
    iZ = static_cast<std::size_t>(floor((p.z - originCorner.z) / voxel_size));
    mapIdx = iX + sampleNX * iY + sampleNX * sampleNY * iZ;

    if (!data.count(mapIdx)) {
      data.emplace(mapIdx, SampledData());
    }

    data[mapIdx].update(p);

    if (!data_choose.count(mapIdx)) {
      data_choose.emplace(mapIdx, SampledListPoints());
    }

    data_choose[mapIdx].update(p, n);
  }

  s_points.reserve(data.size());
  // the average point
  // for (auto& v : data) {
  //   s_points.push_back(v.second.point * (1.0 / v.second.count));
  // }
  // choose the point closest to the average
  PointXYZ closestPoint = PointXYZ();
  PointXYZ closestNormal = PointXYZ();

  for (auto& v : data_choose) {
    v.second.choose(closestPoint, closestNormal);
    s_points.push_back(closestPoint);
    s_normals.push_back(closestNormal);
  }
}

void grid_subsampling_cpu(
  std::vector<PointXYZ>& points,
  std::vector<PointXYZ>& s_points,
  std::vector<long>& lengths,
  std::vector<long>& s_lengths,
  std::vector<PointXYZ>& normals,
  std::vector<PointXYZ>& s_normals,
  float voxel_size
) {
  std::size_t start_index = 0;
  std::size_t batch_size = lengths.size();
  for (std::size_t b = 0; b < batch_size; b++) {
    std::vector<PointXYZ> cur_points = std::vector<PointXYZ>(
      points.begin() + start_index,
      points.begin() + start_index + lengths[b]
    );
    std::vector<PointXYZ> cur_s_points;

    std::vector<PointXYZ> cur_normals = std::vector<PointXYZ>(
      normals.begin() + start_index,
      normals.begin() + start_index + lengths[b]
    );
    std::vector<PointXYZ> cur_s_normals;

    single_grid_subsampling_cpu(cur_points, cur_s_points, cur_normals, cur_s_normals, voxel_size);

    s_points.insert(s_points.end(), cur_s_points.begin(), cur_s_points.end());

    s_normals.insert(s_normals.end(), cur_s_normals.begin(), cur_s_normals.end());

    s_lengths.push_back(cur_s_points.size());

    start_index += lengths[b];
  }

  return;
}
