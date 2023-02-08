import torch
import torch.nn as nn

# from geotransformer.modules.kpconv import nearest_upsample
# from geotransformer.modules.e2pn.blocks_epn import LiftBlockEPN, SimpleBlockEPN, ResnetBottleneckBlockEPN, UnaryBlockEPN, LastUnaryBlockEPN

from geotransformer.modules.e2pn.base_so3conv import preprocess_input, BasicSO3ConvBlock, FinalLinear
from geotransformer.modules.e2pn.vgtk.vgtk.so3conv import get_anchorsV, get_anchors, get_icosahedron_vertices

class E2PNBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, config_epn):
        super(E2PNBackbone, self).__init__()

        # original GeoTransformer backbone strucutre, run out of memory using E2PN
        # mlps=[[init_dim], [init_dim*2], \
        #       [init_dim*2], [init_dim*4], [init_dim*4], \
        #       [init_dim*4], [init_dim*8], [init_dim*8], \
        #       [init_dim*8], [init_dim*16], [init_dim*16]]
        # strides=[1, 1, \
        #          2, 1, 1, \
        #          2, 1, 1, \
        #          2, 1, 1]

        mlps=[[init_dim], [init_dim*2]]
        out_mlps=[mlps[-1][0], output_dim]
        strides=[2, 8]
        initial_radius_ratio = 0.2
        sampling_ratio = 0.8
        sampling_density = 0.4
        kernel_density = 1
        kernel_multiplier = 2
        sigma_ratio= 0.5
        xyz_pooling = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_num = config_epn.num_points
        dropout_rate = 0
        temperature = 3
        so3_pooling = 'attention'
        input_radius = 1 #seems like the input actually has input radius 1?
        kpconv = False

        na = 1 if kpconv else config_epn.kanchor

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

        params['equi2inv'] = {
            'dim_in': mlps[0][0],
            'mlp': [mlps[0][0], output_dim],
            'pooling': so3_pooling,
            'temperature': temperature,
            'kanchor': na,
        }

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

                # one-inter one-intra policy                
                if na == 60:
                    block_type = 'separable_block' 
                elif na == 12:
                    block_type = 'separable_s2_block'
                elif na == 4:
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
    
        params['outblock'] = {
            'dim_in': dim_in,
            'mlp': out_mlps,
            'pooling': so3_pooling,
            'temperature': temperature,
            'kanchor': na,
        }

        # build model
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(BasicSO3ConvBlock(block_param))

        self.outblock = FinalLinear(params['outblock'])

        self.equi2inv = FinalLinear(params['equi2inv'])

        self.na_in = params['na']

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
        # np, 3 -> nb, np, 3
        x = x.unsqueeze(0)
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        x = preprocess_input(x, self.na_in, False)

        for block_i, block in enumerate(self.backbone):
            x = block(x)
            if block_i == 0:
                x_f = self.equi2inv(x)
                points_f = x.xyz.clone().detach().transpose(-1, -2).squeeze(0) # fine points are the points after the first striding
                feat_f = x_f.clone().detach().squeeze(0)

        points_c = x.xyz.clone().detach().transpose(-1, -2).squeeze(0) # coarse points are the last points
        feat_c = x.feats.clone().detach().permute(0, 3, 2, 1).squeeze(0)      

        # TODO: upsample feature for fine matching

        x = self.outblock(x)
        feat_inv_c = x.clone().detach().squeeze(0)

        return feat_c, feat_f, points_c, points_f, feat_inv_c

    def get_anchor(self):
        return self.backbone[-1].get_anchor()