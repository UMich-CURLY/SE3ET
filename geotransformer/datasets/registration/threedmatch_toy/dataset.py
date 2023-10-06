import os.path as osp
import pickle
import random
from typing import Dict

import numpy as np
import torch
import torch.utils.data

from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.registration import get_correspondences
from scipy.spatial.transform import Rotation as R

class ThreeDMatchToySampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
    ):
        super(ThreeDMatchToySampleDataset, self).__init__()

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, 'metadata')
        self.data_root = osp.join(self.dataset_root, 'data')

        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]
            # if subset == 'train':
            #     self.metadata_list =  self.metadata_list[:1]
            # elif subset == 'val':
            #     self.metadata_list =  self.metadata_list[:1]
            self.metadata_list =  self.metadata_list[:100]

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        points = torch.load(osp.join(self.data_root, file_name))
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def _rotate_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""Rotate point clouds.

        ref_points = src_points @ rotation.T
        """
        ref_points = np.matmul(ref_points, rotation.T)

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        r = R.from_euler('z', 0, degrees=True)
        rotation = r.as_matrix()
        translation = np.zeros((3,))

        # get point cloud (ref and src are the same in this toy sample dataset!)
        ref_points = self._load_point_cloud(metadata['pcd0'])
        src_points = self._load_point_cloud(metadata['pcd0'])
        if ref_points.shape[0] > 10000:
            ref_points = ref_points[:20000, :]
            src_points = src_points[:20000, :]

        ref_points, src_points, rotation, translation = self._rotate_point_cloud(
                ref_points, src_points, rotation, translation
            )

        transform = get_transform_from_rotation_translation(rotation, translation)

        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        return data_dict
