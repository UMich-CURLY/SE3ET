import numpy as np
import pypcd


def load_poses(pose_path):
  """ Load ground truth poses (T_w_cam0) from file.
      Args: 
        pose_path: (Complete) filename for the pose file
      Returns: 
        A numpy array of size nx4x4 with n poses as 4x4 transformation 
        matrices
  """
  # Read and parse the poses
  poses = []
  try:
    if '.txt' in pose_path:
      with open(pose_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
          T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
          T_w_cam0 = T_w_cam0.reshape(3, 4)
          T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
          poses.append(T_w_cam0)
    else:
      poses = np.load(pose_path)['arr_0']
  
  except FileNotFoundError:
    print('Ground truth poses are not avaialble.')
  
  return np.array(poses)


def load_calib(calib_path):
  """ Load calibrations (T_cam_velo) from file.
  """
  # Read and parse the calibrations
  T_cam_velo = []
  try:
    with open(calib_path, 'r') as f:
      lines = f.readlines()
      for line in lines:
        if 'Tr:' in line:
          line = line.replace('Tr:', '')
          T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
          T_cam_velo = T_cam_velo.reshape(3, 4)
          T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))
  
  except FileNotFoundError:
    print('Calibrations are not avaialble.')
  
  return np.array(T_cam_velo)

pointcloud = np.load("/home/cel/DockerFolder/code/GeoTransformer/data/Kitti/downsampled/00/000000.npy").reshape((-1,3))
print(pointcloud.shape, '\n', pointcloud)
saved_pointcloud = np.hstack((pointcloud, np.zeros((pointcloud.shape[0], 1))))
submap = pypcd.make_xyz_label_point_cloud(saved_pointcloud)
pcd_file = 'Kitti_00_0.pcd' 
submap.save_pcd(pcd_file)
