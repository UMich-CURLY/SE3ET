import torch
import torch.nn as nn
import numpy as np

# from geotransformer.modules.kpconv import nearest_upsample
# from geotransformer.modules.e2pn.blocks_epn import LiftBlockEPN, SimpleBlockEPN, ResnetBottleneckBlockEPN, UnaryBlockEPN, LastUnaryBlockEPN

from geotransformer.modules.e2pn.base_so3conv import SingleSO3Conv 

from geotransformer.modules.kpconv import UnaryBlock, LastUnaryBlock
from geotransformer.modules.kpconv import nearest_upsample
from geotransformer.utils.memory import gpu_mem_usage, reset_mem_usage

# Local E2PN package
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'vgtk') )
import vgtk.spconv as zptk

class E2PNKPConv(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(E2PNKPConv, self).__init__()

        # GeoTransformer backbone layers
        strides=[1, 1, \
                 1, 1, 1, \
                 1, 1, 1, \
                 1, 1, 1]
        self.dim_in_list = [input_dim, init_dim,
                            init_dim * 2, init_dim * 2,
                            init_dim * 4, init_dim * 4, init_dim * 4,
                            init_dim * 8, init_dim * 8, init_dim * 8,
                            init_dim * 16]        
        self.dim_out_list = [init_dim,
                            init_dim * 2, init_dim * 2,
                            init_dim * 4, init_dim * 4, init_dim * 4,
                            init_dim * 8, init_dim * 8, init_dim * 8,
                            init_dim * 16, init_dim * 16]
        self.radius_list = [init_radius, init_radius, init_radius,
                            init_radius * 2, init_radius * 2, init_radius * 2,                            
                            init_radius * 4, init_radius * 4, init_radius * 4,
                            init_radius * 8, init_radius * 8]
        self.sigma_list = [init_sigma, init_sigma, init_sigma,
                           init_sigma * 2, init_sigma * 2, init_sigma * 2,                            
                           init_sigma * 4, init_sigma * 4, init_sigma * 4,
                           init_sigma * 8, init_sigma * 8]

        self.n_neighbor_list = [26, 26, 26, 31, 31, 31, 33, 33, 33, 37, 37]

        xyz_pooling = None
        kernel_multiplier = 2
        dropout_rate = 0
        na = 1

        params = {'name': 'Equivariant ZPConv Model',
                'backbone': [],
                'equi2inv': [],
                'na': na
                }

        block_param = []
        for i in range(len(self.dim_in_list)):
            lazy_sample = strides[i] == 1

            if i == 0:
                block_type = 'inter_block'
            else:
                block_type = 'inter_residualblock'

            conv_param = {
                'type': block_type,
                'args': {
                    'dim_in': self.dim_in_list[i],
                    'dim_out': self.dim_out_list[i],
                    'kernel_size': kernel_size,
                    'stride': strides[i],
                    'radius': self.radius_list[i],
                    'sigma': self.sigma_list[i],
                    'n_neighbor': self.n_neighbor_list[i],
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier,
                    'activation': 'leaky_relu',
                    'pooling': xyz_pooling,
                    'kanchor': na,
                }
            }
            block_param.append(conv_param)
            
        # build model
        self.encoder1_1 = SingleSO3Conv(block_param[0])
        self.encoder1_2 = SingleSO3Conv(block_param[1])

        self.encoder2_1 = SingleSO3Conv(block_param[2])
        self.encoder2_2 = SingleSO3Conv(block_param[3])
        self.encoder2_3 = SingleSO3Conv(block_param[4])

        self.encoder3_1 = SingleSO3Conv(block_param[5])
        self.encoder3_2 = SingleSO3Conv(block_param[6])
        self.encoder3_3 = SingleSO3Conv(block_param[7])

        self.encoder4_1 = SingleSO3Conv(block_param[8])
        self.encoder4_2 = SingleSO3Conv(block_param[9])
        self.encoder4_3 = SingleSO3Conv(block_param[10])

        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        # separate ref points & src points
        ref_length_s1 = data_dict['lengths'][0][0].item()
        ref_length_s2 = data_dict['lengths'][1][0].item()
        ref_length_s3 = data_dict['lengths'][2][0].item()
        ref_length_s4 = data_dict['lengths'][3][0].item()

        # obtain ref point and src point separately
        # TODO: merge ref and src?
        ref_point_s1 = data_dict['points'][0][:ref_length_s1].unsqueeze(0).permute(0,2,1).contiguous() # [np, 3] -> [nb, 3, np]
        ref_point_s2 = data_dict['points'][1][:ref_length_s2].unsqueeze(0).permute(0,2,1).contiguous() # [np, 3] -> [nb, 3, np]
        ref_point_s3 = data_dict['points'][2][:ref_length_s3].unsqueeze(0).permute(0,2,1).contiguous() # [np, 3] -> [nb, 3, np]
        ref_point_s4 = data_dict['points'][3][:ref_length_s4].unsqueeze(0).permute(0,2,1).contiguous() # [np, 3] -> [nb, 3, np]

        src_point_s1 = data_dict['points'][0][ref_length_s1:].unsqueeze(0).permute(0,2,1).contiguous() # [np, 3] -> [nb, 3, np]
        src_point_s2 = data_dict['points'][1][ref_length_s2:].unsqueeze(0).permute(0,2,1).contiguous() # [np, 3] -> [nb, 3, np]
        src_point_s3 = data_dict['points'][2][ref_length_s3:].unsqueeze(0).permute(0,2,1).contiguous() # [np, 3] -> [nb, 3, np]
        src_point_s4 = data_dict['points'][3][ref_length_s4:].unsqueeze(0).permute(0,2,1).contiguous() # [np, 3] -> [nb, 3, np]

        # obtain ref feat and src subsampling list separately
        ref_subsampling_s2 = subsampling_list[0][:ref_length_s2]
        ref_subsampling_s3 = subsampling_list[1][:ref_length_s3]
        ref_subsampling_s4 = subsampling_list[2][:ref_length_s4]

        src_subsampling_s2 = subsampling_list[0][ref_length_s2:]-ref_length_s1
        src_subsampling_s3 = subsampling_list[1][ref_length_s3:]-ref_length_s2
        src_subsampling_s4 = subsampling_list[2][ref_length_s4:]-ref_length_s3

        # obtain ref feat and src feat separately
        ref_feats_s1 = feats[:ref_length_s1].permute(1,0).unsqueeze(0).unsqueeze(-1) # [np, nc] -> [nb, nc, np, na]
        src_feats_s1 = feats[ref_length_s1:].permute(1,0).unsqueeze(0).unsqueeze(-1)

        # max_mem = gpu_mem_usage()
        # print('max_mem after input', max_mem)
                
        # TODO: input neighbot list as well

        ### ref point ###
        # preprocess input point cloud
        ref_input_s1 = zptk.SphericalPointCloud(ref_point_s1, ref_feats_s1, None)
        
        # stage 1 layers
        ref_input_s1 = self.encoder1_1(ref_input_s1)
        ref_input_s1 = self.encoder1_2(ref_input_s1)
        ref_feats_s1 = ref_input_s1.feats.permute(0, 3, 2, 1).squeeze() # [nb, nc, np, na] -> [np, nc]
        
        # max_mem = gpu_mem_usage()
        # print('max_mem after stage 1', max_mem)
        # stage 2 layers
        ref_input_s2 = self.encoder2_1(ref_input_s1, q_point=ref_point_s2, neighbor_indices=ref_subsampling_s2) # (before downsampling, after downsampling)
        ref_input_s2 = self.encoder2_2(ref_input_s2)
        ref_input_s2 = self.encoder2_3(ref_input_s2)
        ref_feats_s2 = ref_input_s2.feats.permute(0, 3, 2, 1).squeeze() # [nb, nc, np, na] -> [np, nc]
        
        # stage 3 layers
        ref_input_s3 = self.encoder3_1(ref_input_s2, q_point=ref_point_s3, neighbor_indices=ref_subsampling_s3) # (before downsampling, after downsampling)
        ref_input_s3 = self.encoder3_2(ref_input_s3)
        ref_input_s3 = self.encoder3_3(ref_input_s3)
        ref_feats_s3 = ref_input_s3.feats.permute(0, 3, 2, 1).squeeze() # [nb, nc, np, na] -> [np, nc]
        
        # stage 4 layers
        ref_input_s4 = self.encoder4_1(ref_input_s3, q_point=ref_point_s4, neighbor_indices=ref_subsampling_s4) # (before downsampling, after downsampling)
        ref_input_s4 = self.encoder4_2(ref_input_s4)
        ref_input_s4 = self.encoder4_3(ref_input_s4)
        ref_feats_s4 = ref_input_s4.feats.permute(0, 3, 2, 1).squeeze() # [nb, nc, np, na] -> [np, nc]

        ### src point ###
        # preprocess input point cloud
        src_input_s1 = zptk.SphericalPointCloud(src_point_s1, src_feats_s1, None)
        
        # stage 1 layers
        src_input_s1 = self.encoder1_1(src_input_s1)
        src_input_s1 = self.encoder1_2(src_input_s1)
        src_feats_s1 = src_input_s1.feats.permute(0, 3, 2, 1).squeeze() # [nb, nc, np, na] -> [np, nc]
        
        # stage 2 layers
        src_input_s2 = self.encoder2_1(src_input_s1, q_point=src_point_s2, neighbor_indices=src_subsampling_s2) # (before downsampling, after downsampling)
        src_input_s2 = self.encoder2_2(src_input_s2)
        src_input_s2 = self.encoder2_3(src_input_s2)
        src_feats_s2 = src_input_s2.feats.permute(0, 3, 2, 1).squeeze() # [nb, nc, np, na] -> [np, nc]
        
        # stage 3 layers
        src_input_s3 = self.encoder3_1(src_input_s2, q_point=src_point_s3, neighbor_indices=src_subsampling_s3) # (before downsampling, after downsampling)
        src_input_s3 = self.encoder3_2(src_input_s3)
        src_input_s3 = self.encoder3_3(src_input_s3)
        src_feats_s3 = src_input_s3.feats.permute(0, 3, 2, 1).squeeze() # [nb, nc, np, na] -> [np, nc]
        
        # stage 4 layers
        src_input_s4 = self.encoder4_1(src_input_s3, q_point=src_point_s4, neighbor_indices=src_subsampling_s4) # (before downsampling, after downsampling)
        src_input_s4 = self.encoder4_2(src_input_s4)
        src_input_s4 = self.encoder4_3(src_input_s4)
        src_feats_s4 = src_input_s4.feats.permute(0, 3, 2, 1).squeeze() # [nb, nc, np, na] -> [np, nc]

        # combine ref and src features
        feats_s4 = torch.cat((ref_feats_s4, src_feats_s4), 0)
        feats_s3 = torch.cat((ref_feats_s3, src_feats_s3), 0)
        feats_s2 = torch.cat((ref_feats_s2, src_feats_s2), 0)

        latent_s4 = feats_s4
        feats_list.append(feats_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list
