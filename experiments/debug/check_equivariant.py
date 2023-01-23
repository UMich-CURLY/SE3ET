"""
Script to test that features learned from backbone have the equivariance property. 
"""

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from geotransformer.modules.e2pn.base_so3conv import preprocess_input, BasicSO3ConvBlock

from config import make_cfg
cfg = make_cfg()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SO3ConvModel(nn.Module):
    def __init__(self, mlps=[[32]], strides=[1, 1, 1, 1]):
        super(SO3ConvModel, self).__init__()

        mlps=[[32], [64]]
        strides=[1, 1, 1, 1]
        initial_radius_ratio = 0.2
        sampling_ratio = 0.8
        sampling_density = 0.4
        kernel_density = 1
        kernel_multiplier = 2
        sigma_ratio= 0.5
        xyz_pooling = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_num = cfg.num_points 
        dropout_rate = 0
        temperature = 3
        so3_pooling = 'attention'
        input_radius = 0.4
        kpconv = False

        na = 1 if kpconv else cfg.epn.kanchor

        # compute parameters for the model

        # print("[MODEL] USING RADIUS AT %f"%input_radius)
        params = {'name': 'Equivariant ZPConv Model',
                'backbone': [],
                'na': na
                }
        dim_in = 1

        # process args
        n_layer = len(mlps)
        stride_current = 1
        stride_multipliers = [stride_current]
        for i in range(n_layer):
            stride_current *= 2
            stride_multipliers += [stride_current]

        num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]

        radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]

        radii = [r * input_radius for r in radius_ratio]

        # Compute sigma
        weighted_sigma = [sigma_ratio * radii[0]**2]
        for idx, s in enumerate(strides):
            weighted_sigma.append(weighted_sigma[idx] * s)

        for i, block in enumerate(mlps):
            block_param = []
            for j, dim_out in enumerate(block):
                lazy_sample = i != 0 or j != 0

                stride_conv = i == 0 or xyz_pooling != 'stride'

                # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
                neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))

                kernel_size = 1
                if j == 0:
                    # stride at first (if applicable), enforced at first layer
                    inter_stride = strides[i]
                    nidx = i if i == 0 else i+1
                    # nidx = i if (i == 0 or xyz_pooling != 'stride') else i+1
                    if stride_conv:
                        neighbor *= 2 
                        kernel_size = 1 # if inter_stride < 4 else 3
                else:
                    inter_stride = 1
                    nidx = i+1

                # print(f"At block {i}, layer {j}!")
                # print(f'dim_in: {dim_in}')
                # print(f'dim_out: {dim_out}')
                # print(f'neighbor: {neighbor}')
                # print(f'stride: {inter_stride}')
                # print(f'radius : {radii[nidx]}')
                # print(f'sigma : {weighted_sigma[nidx]}')

                # one-inter one-intra policy
                if na == 60:
                    block_type = 'separable_block' 
                elif na == 12:
                    block_type = 'separable_s2_block'
                elif na < 60:
                    block_type = 'inter_block'

                conv_param = {
                    'type': block_type,
                    'args': {
                        'dim_in': dim_in,
                        'dim_out': dim_out,
                        'kernel_size': kernel_size,
                        'stride': inter_stride,
                        'radius': radii[nidx],
                        'sigma': weighted_sigma[nidx],
                        'n_neighbor': neighbor,
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

        # build model
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(BasicSO3ConvBlock(block_param))

        self.na_in = params['na']

    def forward(self, x):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        x = preprocess_input(x, self.na_in, False)

        for block_i, block in enumerate(self.backbone):
            x = block(x)

        output = x.feats.clone().detach()
        # x_equivariant = x.feats.clone().detach().squeeze(0)
        # pcd_downsampled = x.xyz.clone().detach().squeeze(0)
       
        return output

    def get_anchor(self):
        return self.backbone[-1].get_anchor()

def cosine_similarity(x1, x2):
    sum_sim = 0
    same_id = []
    diff_id = []
    for pi in range(min(x1.shape[1], x2.shape[1])):
        curr_sim = torch.absolute(torch.dot(x1[:, pi], x2[:, pi])/(torch.norm(x1[:, pi])*torch.norm(x2[:, pi])))
        sum_sim += curr_sim
        if curr_sim > 0.99:
            same_id.append(pi)
        else:
            diff_id.append(pi)

    avg_sim = sum_sim / x1.shape[1]
    print('same_id', len(same_id), 'diff_id', len(diff_id), 'avg_sim', avg_sim)
    # if len(same_id) > 0:
        # print('x1', x1[same_id[0], :], '\nx2', x2[same_id[0], :])
    # print('diff_id', len(diff_id))
    # if len(diff_id) > 0:
        # print('x1', x1[diff_id[0], :], '\nx2', x2[diff_id[0], :])
    # print('avg_sim', avg_sim)
    return avg_sim

def visualize_points(points, color, vis=False):
    xyz = np.array(points).reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color(color)
    if vis:
        o3d.visualization.draw_geometries([pcd])
    return pcd


if __name__ == '__main__':
    ################
    ## INPUT DATA ##
    ################
    # rotate 72 degree along z axis for kanchor = 60, rotate 120 degree along z axis for kanchor = 12
    if cfg.epn.kanchor == 60:
        theta_1 = torch.Tensor([2 * torch.pi / 5])
    elif cfg.epn.kanchor == 12:
        theta_1 = torch.Tensor([2 * torch.pi / 3])
    rotation_matrix_1 = torch.Tensor([[torch.cos(theta_1), -torch.sin(theta_1), 0],
                                      [torch.sin(theta_1), torch.cos(theta_1), 0],
                                      [0, 0, 1]])
    
    # oxford
    oxford_pointcloud = np.fromfile(str("/home/cel/data/benchmark_datasets/oxford/2014-11-14-16-34-33/pointcloud_20m/1415985079644122.bin"), dtype=np.float64, count=-1).reshape([-1, 3])
    
    # downsample if GPU out of memory
    if cfg.epn.kanchor == 60:
        import random
        downsample_index = random.sample(range(0, 4095), 2048)
        oxford_pointcloud = oxford_pointcloud[downsample_index, :]    
    oxford_pointcloud = torch.Tensor(oxford_pointcloud)

    # rotate 
    rotated_oxford_pointcloud = rotation_matrix_1 @ oxford_pointcloud.T
    rotated_oxford_pointcloud = rotated_oxford_pointcloud.T
    rotated_oxford_pointcloud = rotated_oxford_pointcloud.contiguous()
    
    # visualize oxford point cloud
    # input_oxford_pcd = visualize_points(oxford_pointcloud, [0.5, 0.706, 0], vis=True)
    # visualize rotated oxford point cloud
    # rotated_oxford_pcd = visualize_points(rotated_oxford_pointcloud, [0, 0.706, 1], vis=True)

    #############
    ## NETWORK ##
    #############
    one_layer_encoder = SO3ConvModel()
    print(one_layer_encoder)
    one_layer_encoder = one_layer_encoder.to(device)

    #####################
    ## precompute data ##
    #####################
    network_input_points = oxford_pointcloud.unsqueeze(0).to(device) # [nb, np, 3]
    rotated_network_input_points = rotated_oxford_pointcloud.unsqueeze(0).to(device) # [nb, np, 3]    

    ##########################
    ## EQUIVARIANT FEATURES ##
    ##########################    
    oxford_one_layer_feats = one_layer_encoder(network_input_points) # nb, 32, np, na
    oxford_one_layer_feats = oxford_one_layer_feats.detach()[0] # 32, np, na

    rotated_oxford_one_layer_feats = one_layer_encoder(rotated_network_input_points) # nb, 32, np, na
    rotated_oxford_one_layer_feats = rotated_oxford_one_layer_feats.detach()[0] # 32, np, na
    print('one_layer_feats', oxford_one_layer_feats.shape)
    print('rotated_oxford_one_layer_feats', rotated_oxford_one_layer_feats.shape)

    for anchor_i in range(cfg.epn.kanchor):
        print('similarity_one_layer_0_'+str(anchor_i))
        similarity = cosine_similarity(oxford_one_layer_feats[:, :, 0], rotated_oxford_one_layer_feats[:, :, anchor_i])

    # print('\n==== feature debug ====')
    # print('one_layer_feats\n', one_layer_feats[:5, 0, :10])
    # print('rotated_one_layer_feats 0\n', rotated_one_layer_feats[:5, 0, :10])
    # print('rotated_one_layer_feats 1\n', rotated_one_layer_feats[:5, 1, :10])
    # print('rotated_one_layer_feats 2\n', rotated_one_layer_feats[:5, 2, :10])
    