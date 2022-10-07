import torch
import torch.nn as nn
from IPython import embed

from geotransformer.modules.kpconv import nearest_upsample
from geotransformer.modules.e2pn.blocks_epn import SimpleBlockEPN, ResnetBottleneckBlockEPN, UnaryBlockEPN, LastUnaryBlockEPN
class E2PN(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, config_epn):
        super(E2PN, self).__init__()

        radius = config_epn.first_subsampling_dl * config_epn.conv_radius        

        self.encoder1_1 = SimpleBlockEPN('simple_epn', input_dim, init_dim, radius, 0, config_epn)
        self.encoder1_2 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim, init_dim * 2, radius, 1, config_epn) #resnetb_strided_epn?

        self.encoder2_1 = ResnetBottleneckBlockEPN('resnetb_strided_epn', init_dim * 2, init_dim * 2, radius, 2, config_epn)
        self.encoder2_2 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 2, init_dim * 4, radius, 3, config_epn)
        self.encoder2_3 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 4, init_dim * 4, radius, 4, config_epn)

        self.encoder3_1 = ResnetBottleneckBlockEPN('resnetb_strided_epn', init_dim * 4, init_dim * 4, radius, 5, config_epn)
        self.encoder3_2 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 4, init_dim * 8, radius, 6, config_epn)
        self.encoder3_3 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 8, init_dim * 8, radius, 7, config_epn)

        self.encoder4_1 = ResnetBottleneckBlockEPN('resnetb_strided_epn', init_dim * 8, init_dim * 8, radius, 8, config_epn)
        self.encoder4_2 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 8, init_dim * 16, radius, 9, config_epn)
        self.encoder4_3 = ResnetBottleneckBlockEPN('resnetb_epn', init_dim * 16, init_dim * 16, radius, 10, config_epn)

        self.decoder3 = UnaryBlockEPN(init_dim * 24, init_dim * 8, config_epn.use_batch_norm, config_epn.batch_norm_momentum)
        self.decoder2 = LastUnaryBlockEPN(init_dim * 12, output_dim)

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        print('input feature', feats.shape)
        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        print('feats_s1', feats_s1.shape)

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        print('feats_s2', feats_s2.shape)

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        print('feats_s3', feats_s3.shape)

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        print('feats_s4', feats_s4.shape)

        latent_s4 = feats_s4
        feats_list.append(feats_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)
        print('latent_s3', latent_s3.shape)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)
        print('latent_s2', latent_s2.shape)

        feats_list.reverse()

        return feats_list