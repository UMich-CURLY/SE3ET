#include <cstring>
#include "grid_subsampling.h"
#include "grid_subsampling_cpu.h"

std::vector<at::Tensor> grid_subsampling(
  at::Tensor points,
  at::Tensor lengths,
  at::Tensor normals,
  float voxel_size
) {
  CHECK_CPU(points);
  CHECK_CPU(lengths);
  CHECK_CPU(normals);
  CHECK_IS_FLOAT(points);
  CHECK_IS_LONG(lengths);
  CHECK_IS_FLOAT(normals);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(lengths);
  CHECK_CONTIGUOUS(normals);

  std::size_t batch_size = lengths.size(0);
  std::size_t total_points = points.size(0);

  std::vector<PointXYZ> vec_points = std::vector<PointXYZ>(
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()),
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()) + total_points
  );
  std::vector<PointXYZ> vec_s_points;

  std::vector<long> vec_lengths = std::vector<long>(
    lengths.data_ptr<long>(),
    lengths.data_ptr<long>() + batch_size
  );
  std::vector<long> vec_s_lengths;

  std::vector<PointXYZ> vec_normal = std::vector<PointXYZ>(
    reinterpret_cast<PointXYZ*>(normals.data_ptr<float>()),
    reinterpret_cast<PointXYZ*>(normals.data_ptr<float>()) + total_points
  );
  std::vector<PointXYZ> vec_s_normals;

  grid_subsampling_cpu(
    vec_points,
    vec_s_points,
    vec_lengths,
    vec_s_lengths,
    vec_normal,
    vec_s_normals,
    voxel_size
  );

  std::size_t total_s_points = vec_s_points.size();
  at::Tensor s_points = torch::zeros(
    {total_s_points, 3},
    at::device(points.device()).dtype(at::ScalarType::Float)
  );
  at::Tensor s_lengths = torch::zeros(
    {batch_size},
    at::device(lengths.device()).dtype(at::ScalarType::Long)
  );
  at::Tensor s_normals = torch::zeros(
    {total_s_points, 3},
    at::device(normals.device()).dtype(at::ScalarType::Float)
  );

  std::memcpy(
    s_points.data_ptr<float>(),
    reinterpret_cast<float*>(vec_s_points.data()),
    sizeof(float) * total_s_points * 3
  );
  std::memcpy(
    s_lengths.data_ptr<long>(),
    vec_s_lengths.data(),
    sizeof(long) * batch_size
  );
  std::memcpy(
    s_normals.data_ptr<float>(),
    reinterpret_cast<float*>(vec_s_normals.data()),
    sizeof(float) * total_s_points * 3
  );

  return {s_points, s_lengths, s_normals};
}
