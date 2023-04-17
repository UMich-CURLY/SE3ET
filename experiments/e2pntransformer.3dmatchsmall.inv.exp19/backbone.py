import torch
import torch.nn as nn

from geotransformer.modules.kpconv import nearest_upsample
from geotransformer.modules.e2pn.blocks_epn import LiftBlockEPN, SimpleBlockEPN, ResnetBottleneckBlockEPN, UnaryBlockEPN, LastUnaryBlockEPN, InvOutBlockEPN
from geotransformer.modules.kpconv import UnaryBlock, LastUnaryBlock, nearest_upsample

class E2PN(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, init_radius, init_sigma, group_norm, config_epn):
        super(E2PN, self).__init__()
        self.preprocess = LiftBlockEPN('lift_epn', input_dim, config_epn)

        self.encoder1_1 = SimpleBlockEPN('simple_epn', input_dim, init_dim, init_radius, init_sigma, config_epn)
        self.encoder1_2 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim, init_dim * 2, init_radius, init_sigma, config_epn) #resnetb_strided_epn?

        self.encoder2_1 = ResnetBottleneckBlockEPN('resnetb_strided_epn', init_dim * 2, init_dim * 2, init_radius, init_sigma, config_epn)
        self.encoder2_2 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 2, init_dim * 4, init_radius * 2, init_sigma * 2, config_epn)
        self.encoder2_3 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 4, init_dim * 4, init_radius * 2, init_sigma * 2, config_epn)
        self.equ2inv2 = InvOutBlockEPN('inv_epn', init_dim * 4, config_epn)

        self.encoder3_1 = ResnetBottleneckBlockEPN('resnetb_strided_epn', init_dim * 4, init_dim * 4, init_radius * 2, init_sigma * 2, config_epn)
        self.encoder3_2 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 4, init_dim * 8, init_radius * 4, init_sigma * 4, config_epn)
        self.encoder3_3 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 8, init_dim * 8, init_radius * 4, init_sigma * 4, config_epn)
        self.equ2inv3 = InvOutBlockEPN('inv_epn', init_dim * 8, config_epn)

        self.encoder4_1 = ResnetBottleneckBlockEPN('resnetb_strided_epn', init_dim * 8, init_dim * 8, init_radius * 4, init_sigma * 4, config_epn)
        self.encoder4_2 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 8, init_dim * 16, init_radius * 8, init_sigma * 8, config_epn)
        self.encoder4_3 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 16, init_dim * 16, init_radius * 8, init_sigma * 8, config_epn)
        self.equ2inv4 = InvOutBlockEPN('inv_epn', init_dim * 16, config_epn)
                                       
        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        feats_s1 = feats # N x 1 # points_list[0]: N x 3
        feats_s1 = self.preprocess(feats_s1) # N x kanchor x 1
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0]) # N x kanchor x init_dim
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0]) # N x kanchor x init_dim*2

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0]) # 6460 x kanchor x init_dim*2
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1]) # 6460 x kanchor x init_dim*4
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1]) # 6460 x kanchor x init_dim*4
        feats_s2_inv = self.equ2inv2(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2]) # 1844 x kanchor x init_dim*8
        feats_s3_inv = self.equ2inv3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3]) # 549 x kanchor x init_dim*16
        feats_s4_inv = self.equ2inv4(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        latent_s4 = feats_s4_inv
        feats_list.append(feats_s4_inv)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3_inv], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2_inv], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list