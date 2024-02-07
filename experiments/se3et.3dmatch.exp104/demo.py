import argparse

import torch
import numpy as np

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--ref_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--gt_file", required=True, help="ground-truth transformation file")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args):
    src_points = np.load(args.src_file)
    ref_points = np.load(args.ref_file)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)

    return data_dict


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)
    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]
    transform = data_dict["transform"]

    # visualization
    print("raw input")
    ref_pcd = make_open3d_point_cloud(ref_points)
    ref_pcd.estimate_normals()
    ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    src_pcd.paint_uniform_color(get_color("custom_blue"))
    draw_geometries(ref_pcd, src_pcd)

    print("gt")
    src_pcd_gt = src_pcd
    src_pcd_gt = src_pcd_gt.transform(transform)
    draw_geometries(ref_pcd, src_pcd_gt)

    print("estimation")
    src_pcd_est = make_open3d_point_cloud(src_points)
    src_pcd_est.estimate_normals()
    src_pcd_est.paint_uniform_color(get_color("custom_blue"))
    src_pcd_est = src_pcd_est.transform(estimated_transform)
    draw_geometries(ref_pcd, src_pcd_est)

    # compute error
    rre, rte = compute_registration_error(transform, estimated_transform)
    print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")

    #===========================
    # rotated
    #===========================

    from scipy.spatial.transform import Rotation
    axis = np.random.rand(3) - 0.5
    axis = axis / np.linalg.norm(axis) + 1e-8
    theta = np.pi * np.random.rand()
    euler = axis * theta
    src_rotation = Rotation.from_euler('zyx', euler).as_matrix()

    src_points_rotated = np.matmul(src_points, src_rotation.T)
    transform_rotated = np.vstack((np.hstack((np.matmul(transform[:3, :3], src_rotation.T), transform[:3, 3].reshape((3,1)))), np.array([0, 0, 0, 1])))
    
    # prepare data
    data_dict = load_data(args)
    data_dict["src_points"] = src_points_rotated.astype(np.float32)
    data_dict["src_feats"] = np.ones_like(src_points[:, :1]).astype(np.float32)
    data_dict["transform"] = transform_rotated.astype(np.float32)
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    estimated_transform_rotated = output_dict["estimated_transform"]
    
    print("rotated raw")
    src_pcd_rotated = make_open3d_point_cloud(src_points_rotated)
    src_pcd_rotated.estimate_normals()
    src_pcd_rotated.paint_uniform_color(get_color("custom_blue"))
    draw_geometries(ref_pcd, src_pcd_rotated)

    print("rotated gt")
    src_pcd_gt_rotated = src_pcd_rotated
    src_pcd_gt_rotated = src_pcd_gt_rotated.transform(transform_rotated)
    draw_geometries(ref_pcd, src_pcd_gt_rotated)

    print("rotated estimation")
    src_pcd_est_rotated = make_open3d_point_cloud(src_points_rotated)
    src_pcd_est_rotated.estimate_normals()
    src_pcd_est_rotated.paint_uniform_color(get_color("custom_blue"))
    src_pcd_est_rotated = src_pcd_est_rotated.transform(estimated_transform_rotated)
    draw_geometries(ref_pcd, src_pcd_est_rotated)

    # compute error
    rre, rte = compute_registration_error(transform_rotated, estimated_transform_rotated)
    print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")


if __name__ == "__main__":
    main()
