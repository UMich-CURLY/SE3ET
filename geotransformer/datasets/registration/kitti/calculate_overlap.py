import numpy as np

from config import make_cfg
from geotransformer.datasets.registration.kitti.dataset import OdometryKittiPairDataset
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)
from scipy.spatial import cKDTree
import open3d as o3d
from tqdm import tqdm

def train_valid_data_loader(cfg, distributed):
    train_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_min_scale=cfg.train.augmentation_min_scale,
        augmentation_max_scale=cfg.train.augmentation_max_scale,
        augmentation_shift=cfg.train.augmentation_shift,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg):
    train_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_min_scale=cfg.train.augmentation_min_scale,
        augmentation_max_scale=cfg.train.augmentation_max_scale,
        augmentation_shift=cfg.train.augmentation_shift,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_root,
        'test',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits

def calculate_overlapping_rate(point_cloud1, point_cloud2, threshold):
    """
    Calculate the overlapping rate of two point clouds.
    
    Parameters:
    - point_cloud1: numpy array of shape (N, 3)
    - point_cloud2: numpy array of shape (M, 3)
    - threshold: float, the distance threshold to consider points as overlapping
    
    Returns:
    - overlapping_rate: float, the overlapping rate
    """
    # Build KDTree for fast nearest-neighbor lookup
    tree1 = cKDTree(point_cloud1)
    tree2 = cKDTree(point_cloud2)
    
    # Find the nearest neighbors in point_cloud2 for each point in point_cloud1
    distances1, _ = tree1.query(point_cloud2, distance_upper_bound=threshold)
    distances2, _ = tree2.query(point_cloud1, distance_upper_bound=threshold)
    
    # Count overlapping points
    overlap1 = np.sum(distances1 <= threshold)
    overlap2 = np.sum(distances2 <= threshold)
    
    # Calculate overlapping rate
    overlapping_rate1 = overlap1 / point_cloud1.shape[0]
    overlapping_rate2 = overlap2 / point_cloud2.shape[0]
    
    # Return the average overlapping rate
    overlapping_rate = (overlapping_rate1 + overlapping_rate2) / 2
    return overlapping_rate


def get_overlap_ratio(source, target, threshold=0.03):
    """
    We compute overlap ratio from source point cloud to target point cloud
    code from https://github.com/prs-eth/OverlapPredator/blob/main/scripts/cal_overlap.py
    """
    pcd_tree = o3d.geometry.KDTreeFlann(target)
    
    match_count=0
    for i, point in enumerate(source.points):
        [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if(count!=0):
            match_count+=1

    overlap_ratio = match_count / len(source.points)
    return overlap_ratio


def visualize_pointcloud_pairs(np_ref_points, np_transformed_src_points, idx, overlap):
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(np_ref_points)
    ref_pcd.paint_uniform_color([1, 0.706, 0])

    transformed_src_pcd = o3d.geometry.PointCloud()
    transformed_src_pcd.points = o3d.utility.Vector3dVector(np_transformed_src_points)
    transformed_src_pcd.paint_uniform_color([0, 0.706, 1])

    # visualize point cloud pairs
    # o3d.visualization.draw_geometries([ref_pcd, transformed_src_pcd])

    # save image instead of showing
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(ref_pcd)
    vis.add_geometry(transformed_src_pcd)
    vis.update_geometry(ref_pcd)
    vis.update_geometry(transformed_src_pcd)
    vis.poll_events()
    vis.update_renderer()
    overlap_string = f"{overlap:.3f}"
    vis.capture_screen_image('kitti_pair_' + str(idx) + '_overlap_' + overlap_string + '.png', do_render=True)
    vis.destroy_window()


def main(mode='train'):
    # create data loader
    cfg = make_cfg()
    train_loader, val_loader, _ = train_valid_data_loader(cfg, False)
    test_loader, _ = test_data_loader(cfg)

    # choose data loader
    if mode == 'train':
        data_loader = train_loader
    elif mode == 'val':
        data_loader = val_loader
    elif mode == 'test':
        data_loader = test_loader

    overlap_list = []

    for iteration, data_dict in tqdm(enumerate(data_loader)):
        # load data from each iteration
        ref_length = data_dict['lengths'][0][0].item()
        points = data_dict['points'][0].detach()
        ref_points = points[:ref_length]
        src_points = points[ref_length:]
        transform = data_dict['transform'].detach() # ref_points = src_points @ rotation.T + translation
        # print('transformation matrix\n', transform)

        # point clouds in numpy for computation        
        np_ref_points = np.array(ref_points).astype(np.float32)
        np_src_points = np.array(src_points).astype(np.float32)

        # transformer src point cloud
        np_transformed_src_points = transform @ np.vstack((np_src_points.T, np.ones((1, np_src_points.shape[0])))) # 4, M
        np_transformed_src_points = np_transformed_src_points.T # M, 4
        np_transformed_src_points = np.array(np_transformed_src_points)[:, :3] 
        
        # calculate overlap rate
        threshold = 0.3  # Adjust threshold as needed, 0.05 for 3DMatch, 0.3 for KITTI
        overlapping_rate = calculate_overlapping_rate(np_ref_points, np_transformed_src_points, threshold)

        # second way of calculating overlap
        # ref_pcd = o3d.geometry.PointCloud()
        # ref_pcd.points = o3d.utility.Vector3dVector(np_ref_points)
        # transformed_src_pcd = o3d.geometry.PointCloud()
        # transformed_src_pcd.points = o3d.utility.Vector3dVector(np_transformed_src_points)
        # overlap2 = get_overlap_ratio(ref_pcd, transformed_src_pcd, threshold)
        # print('overlap', overlapping_rate, overlap2)
        
        if iteration < 50:
            visualize_pointcloud_pairs(np_ref_points, np_transformed_src_points, iteration, overlapping_rate)

        # store in the list
        overlap_list.append(overlapping_rate)

    np_overlap = np.array(overlap_list)
    avg_overlap = np.mean(np_overlap)
    std_overlap = np.std(np_overlap)
    print('avg_overlap', avg_overlap)
    print('std_overlap', std_overlap)





if __name__ == '__main__':
    main(mode='train')