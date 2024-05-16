import importlib


ext_module = importlib.import_module('geotransformer.ext')


def grid_subsample(points, lengths, normals, voxel_size):
    """Grid subsampling in stack mode.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        normals (Tensor): stacked normals. (N, 3)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
        s_normals (Tensor): stacked subsampled normals (N, 3)
    """
    s_points, s_lengths, s_normals = ext_module.grid_subsampling(points, lengths, normals, voxel_size)
    return s_points, s_lengths, s_normals
