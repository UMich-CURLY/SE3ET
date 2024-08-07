from functools import partial

import numpy as np
import torch

from geotransformer.modules.ops import grid_subsample, radius_search
from geotransformer.utils.torch import build_dataloader
from geotransformer.utils.open3d import estimate_normals

# Stack mode utilities


def precompute_data_stack_mode(points, lengths, num_stages, voxel_size, radius, neighbor_limits):
    assert num_stages == len(neighbor_limits)

    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []
    normals_list = []

    # estimate normals for initial point clouds
    ref_point_c = points[:lengths[0]]
    src_point_c = points[lengths[0]:]
    ref_normal_c = torch.from_numpy(estimate_normals(ref_point_c))
    src_normal_c = torch.from_numpy(estimate_normals(src_point_c))
    normals = torch.cat((ref_normal_c, src_normal_c), 0).float()

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, lengths, normals = grid_subsample(points, lengths, normals, voxel_size=voxel_size)
        if i == num_stages - 1:
            # maximum 2000 points for the final stage
            if lengths[0] > 2000:
                points = torch.cat((points[:2000], points[lengths[0]:]), dim=0)
                normals = torch.cat((normals[:2000], normals[lengths[0]:]), dim=0)
                lengths[0] = 2000
            if lengths[1] > 2000:
                points = torch.cat((points[:lengths[0]], points[lengths[0]:lengths[0]+2000]), dim=0)
                normals = torch.cat((normals[:lengths[0]], normals[lengths[0]:lengths[0]+2000]), dim=0)
                lengths[1] = 2000
        points_list.append(points)
        lengths_list.append(lengths)
        normals_list.append(normals)
        voxel_size *= 2
       

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]
        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                radius * 2,
                neighbor_limits[i + 1],
            )
            upsampling_list.append(upsampling)

        radius *= 2

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
        'normals': normals_list,
    }


def single_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for single point cloud in stack mode.

    Points are organized in the following order: [P_1, ..., P_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool=True)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: feats, points, normals
    if 'normals' in collated_dict:
        normals = torch.cat(collated_dict.pop('normals'), dim=0)
    else:
        normals = None
    feats = torch.cat(collated_dict.pop('feats'), dim=0)
    points_list = collated_dict.pop('points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    if normals is not None:
        collated_dict['normals'] = normals
    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True, use_normal=False
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)
    
    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000, use_normal=False
):
    # Compute higher bound of neighbors number in a neighborhood
    # calculate number of voxels in this search radius volumn
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3)) 
    # create an empty histogram
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    # set the highest number of neighbor we want to find for each layer as hist_n
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True, use_normal=use_normal
        )

        # update histogram
        # calculate number of neighbors per point (N, 1) for each stage 
        # neighbors < N is used because radius search would filled with N if there are less than max_neighbor_limits
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        # create neighbot histogram (hist_n, 1) for each stage
        # element in index i represents number of points that has i neighbors
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        # stack neighbor histograms for each stage together, (num_stages, hist_n)
        # keep adding the statistics from different point clouds
        neighbor_hists += np.vstack(hists)

        # stop calculating when we have enough samples for all the stages
        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break
    
    # calculate accumulated distribution for each stage, (hist_n, num_stages)
    # element in index (i, n) means the number of point has number of neighbors smaller or equal to i in stage n
    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    # find the limit for number of neighbors using keep ratio times the total number of points in the statistics
    # for example, if we have 5k points in the histogram and the keep ratio is 0.8,
    # then 4k points has number of neighbors smaller than neighbor_limits 
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    search_radius,
    neighbor_limits,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,
    precompute_data=True,
    use_normal=False,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=precompute_data,
            use_normal=use_normal,
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader
