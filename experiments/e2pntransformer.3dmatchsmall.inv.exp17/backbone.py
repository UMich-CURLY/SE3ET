import torch
import torch.nn as nn
import numpy as np

# from geotransformer.modules.kpconv import nearest_upsample
# from geotransformer.modules.e2pn.blocks_epn import LiftBlockEPN, SimpleBlockEPN, ResnetBottleneckBlockEPN, UnaryBlockEPN, LastUnaryBlockEPN

from geotransformer.modules.e2pn.base_so3conv import preprocess_input, BasicSO3ConvBlock, FinalLinear, InvariantPooling
from geotransformer.modules.e2pn.vgtk.vgtk.so3conv import get_anchorsV, get_anchors, get_icosahedron_vertices

from geotransformer.modules.kpconv import UnaryBlock, LastUnaryBlock
from geotransformer.modules.ops import radius_search
from geotransformer.modules.kpconv import nearest_upsample

class E2PNBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, group_norm, config_epn):
        super(E2PNBackbone, self).__init__()

        # GeoTransformer backbone layers
        mlps=[init_dim, init_dim*2, \
              init_dim*2, init_dim*4, init_dim*4, \
              init_dim*4, init_dim*8, init_dim*8, \
              init_dim*8, init_dim*16, init_dim*16]
        strides=[1, 1, \
                 3, 1, 1, \
                 3, 1, 1, \
                 3, 1, 1]
        self.firststride = next(x for x, val in enumerate(strides) if val > 1)
        self.storelayer = [1, 4, 7, 10]

        # sampling params
        self.neighbor_limits = []
        self.radius_list = []

        # params for radius, sigma, and neighbors
        input_num = config_epn.num_points
        input_radius = 2.5 # radius of the input point cloud, change according to input dataset
        initial_radius_ratio = 0.025 # set to same as GeoTRansformer's voxel size
        sampling_ratio = 0.8 # keep ratio for neighbors
        sigma_ratio = 2.0
        xyz_pooling = None

        kernel_multiplier = 2
        kernel_size = 1
        dropout_rate = 0
        temperature = 3
        so3_pooling = 'attention'
        kpconv = True

        na = 1 if kpconv else config_epn.kanchor

        # compute parameters for the model

        params = {'name': 'Equivariant ZPConv Model',
                'backbone': [],
                'equi2inv': [],
                'na': na
                }
        dim_in = 1

        # accumulated multiply stride
        radius_multipliers = []
        stride_cumsum = []
        stride_current = 1
        radius_current = 1
        for idx, s in enumerate(strides):
            radius_multipliers.append(radius_current)
            stride_cumsum.append(stride_current)
            if s > 1:
                radius_current *= 2
                stride_current *= s

        # number of points in each layer
        num_centers = [int(input_num / multiplier) for multiplier in stride_cumsum]

        # search radius in each layer, double the radius when downsampled
        self.radius_list = [input_radius * initial_radius_ratio * multiplier for multiplier in radius_multipliers]

        # Compute sigma
        self.weighted_sigma = [sigma_ratio * initial_radius_ratio * multiplier for multiplier in radius_multipliers]

        # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
        # hard coded for now:
        # assume num_centers[i] are equally distributed in 0.4 * input_radius volumn, 
        # calculate the number of points that lies in radius_ratio[i] by ratio
        # self.neighbor_limits = [max(20, int(sampling_ratio * num_centers[idx] * (self.radius_list[idx] / (0.4 * input_radius))**3)) for idx in range(len(mlps))]
        self.neighbor_limits = [75, 75, 75, 100, 100, 100, 125, 125, 125, 150, 150]

        for i, dim_out in enumerate(mlps):
            block_param = []
            lazy_sample = strides[i] == 1

            # one-inter one-intra policy                
            if na == 60:
                block_type = 'separable_block' 
            elif na == 12:
                if i == 0:
                    block_type = 'separable_s2_block'
                block_type = 'separable_s2_residualblock'
            elif na == 4:
                if i == 0:
                    block_type = 'separable_s2_block'
                block_type = 'separable_s2_residualblock'
            elif na < 60:
                block_type = 'inter_block'
                if i == 0:
                    block_type = 'inter_block'
                block_type = 'inter_residualblock'

            conv_param = {
                'type': block_type,
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size,
                    'stride': strides[i],
                    'radius': self.radius_list[i],
                    'sigma': self.weighted_sigma[i],
                    'n_neighbor': self.neighbor_limits[i],
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier,
                    'activation': 'leaky_relu',
                    'pooling': xyz_pooling,
                    'kanchor': na,
                }
            }
            block_param.append(conv_param)
            
            dim_in = dim_out

            params['backbone'].append(block_param)
            
            if i in self.storelayer:
                if i == (len(mlps)-1):
                    dim_out = output_dim

                linear_param = {
                    'dim_in': dim_in,
                    'mlp': dim_out,
                    'pooling': so3_pooling,
                    'temperature': temperature,
                    'kanchor': na,
                }
                
                params['equi2inv'].append(linear_param)

        print('num_centers', num_centers)
        print('radius', self.radius_list)
        print('sigma', self.weighted_sigma)
        print('neighbor limits', self.neighbor_limits)

        # build model
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(BasicSO3ConvBlock(block_param))
        
        self.equi2inv = nn.ModuleList()
        for inv_param in params['equi2inv']:
            self.equi2inv.append(InvariantPooling(inv_param))

        self.na_in = params['na']
        
        self.decoder3 = UnaryBlock(mlps[self.storelayer[-1]]+mlps[self.storelayer[-2]], mlps[self.storelayer[-2]], group_norm)
        self.decoder2 = LastUnaryBlock(mlps[self.storelayer[-2]]+mlps[self.storelayer[-3]], output_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): (N, 3) one input pointcloud
        Returns:
            feat_c (Tensor): (A, N'', C) output features
            feat_f (Tensor): (N', C) output finer invariant features for fine matching
            points_c (Tensor): (N'', 3) output features
            points_f (Tensor): (N', 3) output features
            feat_inv_c (Tensor): (N'', C) output features
        """
        feature_list = []
        point_list = []
        sample_idx_list = []
        inv_feat_list = []

        # preprocess input point cloud
        # np, 3 -> nb, np, 3
        x = x.unsqueeze(0)
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        x = preprocess_input(x, self.na_in, False)

        # learn equivariant features
        for block_i, block in enumerate(self.backbone):
            x, sample_idx = block(x)
            if block_i in self.storelayer:
                feature_list.append(x)
                point_list.append(x.xyz.clone().detach().transpose(-1, -2).squeeze(0))
                sample_idx_list.append(sample_idx.squeeze(0).long())
            if block_i == self.firststride:
                points_f = x.xyz.clone().detach().transpose(-1, -2).squeeze(0) # fine points are the points after the first striding

        points_c = x.xyz.clone().detach().transpose(-1, -2).squeeze(0) # coarse points are the points from last layer
        feat_c = x.feats.clone().detach().permute(0, 3, 2, 1).squeeze(0)

        # extract invariant features from equivariant features
        for inv_i, inv in enumerate(self.equi2inv):
            inv_x = inv(feature_list[inv_i])
            inv_feat_list.append(inv_x.squeeze(0))

        feat_inv_c = inv_feat_list[-1]
        point_s4 = torch.Tensor(point_list[-1].clone().detach().cpu()).contiguous()
        point_s3 = torch.Tensor(point_list[-2].clone().detach().cpu()).contiguous()
        point_s2 = torch.Tensor(point_list[-3].clone().detach().cpu()).contiguous()

        # upsample feature for fine matching
        # make last layer feature as same number of points as the upper layer
        upsampling_s4 = radius_search(point_s3, 
                                      point_s4, 
                                      torch.Tensor([point_s3.shape[0]]).long(), 
                                      torch.Tensor([point_s4.shape[0]]).long(), 
                                      self.radius_list[-1],
                                      self.neighbor_limits[-1],)
        upsampling_s4 = upsampling_s4.to(feat_inv_c.device)
        latent_s3 = nearest_upsample(feat_inv_c, upsampling_s4)
        latent_s3 = torch.cat([latent_s3, inv_feat_list[-2]], dim=1)
        latent_s3 = self.decoder3(latent_s3)

        upsampling_s3 = radius_search(point_s2, 
                                      point_s3, 
                                      torch.Tensor([point_s2.shape[0]]).long(), 
                                      torch.Tensor([point_s3.shape[0]]).long(), 
                                      self.radius_list[-2],
                                      self.neighbor_limits[-2],)
        upsampling_s3 = upsampling_s3.to(latent_s3.device)
        latent_s2 = nearest_upsample(latent_s3, upsampling_s3)
        latent_s2 = torch.cat([latent_s2, inv_feat_list[-3]], dim=1)
        latent_s2 = self.decoder2(latent_s2)      
        feat_f = latent_s2

        return feat_c, feat_f, points_c, points_f, feat_inv_c

    def get_anchor(self):
        return self.backbone[-1].get_anchor()
