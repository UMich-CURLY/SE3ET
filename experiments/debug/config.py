import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.init_voxel_size = 0.025 # 0.0025 for oxford # 0.025 for 3d match
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256


# epn
_C.epn = edict()
# ======== SO(2) ======== 
### Valid parameters: kanchor < 10, quotient_factor >= 1, k = 15
# _C.epn.kanchor = 3 # number of rotation anchors A; 12: E2PN, 60 & lower than 10: EPN
# _C.epn.quotient_factor = 1 # > 1 enables quotient features. 2 results in C6/C2. 
# _C.epn.num_kernel_points = 15 # Number of kernel points
# # ======== E2PN ======== 
# ### Valid parameter set 1: kanchor = 12, quotient_factor  = 5, k = 13
# _C.epn.kanchor = 12 # number of rotation anchors A; 12: E2PN, 60 & lower than 10: EPN
# _C.epn.quotient_factor = 5 # > 1 enables quotient features. 2 results in C6/C2. 
# _C.epn.num_kernel_points = 13 # Number of kernel points
# _C.num_points = 4096
# ======== E2PN ======== 
### Valid parameter set 1: kanchor = 4, quotient_factor  = 3, k = 15
_C.epn.kanchor = 4 # number of rotation anchors A; 12: E2PN, 60 & lower than 10: EPN
_C.epn.quotient_factor = 3 # > 1 enables quotient features. 2 results in C6/C2. 
_C.epn.num_kernel_points = 15 # Number of kernel points
_C.num_points = 4096
# ======== EPN ======== 
# ### Valid parameter set 2: kanchor = 60, quotient_factor  = 1, k = 13
# _C.epn.kanchor = 60 # number of rotation anchors A; 12: E2PN, 60 & lower than 10: EPN
# _C.epn.quotient_factor = 1 # > 1 enables quotient features. 2 results in C6/C2. 
# _C.epn.num_kernel_points = 13 # Number of kernel points
# _C.num_points = 2048
# ========
_C.epn.epn_kernel = False  # whether or not use the kernel file from epn # always false
_C.epn.att_pooling = False # early fusion with attentive averge pooling
_C.epn.att_permute = False # early fusion with soft permutation
_C.epn.dual_feature = False # use both permutation and max pooling as the invariant layer
_C.epn.ctrness_w_track = False # use the max pooling feature to regress centerness noise free kernel
_C.epn.equiv_mode_kp = True # False: for KPConv, load kernel with small noise # combined with fixed_kernel_points = 'verticals' results in z-axis symmetric kernel
_C.epn.non_sep_conv = False # full (non-separable) conv; True: E2PN, False: EPN
_C.epn.rot_by_permute = False # rotate-by-permute in full conv
_C.epn.rot_head_attn = False # late fusion with attention weight. Need to remove 'inv_epn' to enable late fusion. 
_C.epn.ignore_steer_constraint = False # ignore steerability constraint when using quotient features
_C.epn.gather_by_idxing = False # efficient gathering, False for default, True for faster
_C.epn.use_batch_norm = True # Batch normalization parameters
_C.epn.batch_norm_momentum = 0.99 # Batch normalization parameters
_C.epn.first_subsampling_dl = 0.02 # Size of the first subsampling grid in meter
_C.epn.conv_radius = 2.5 # Radius of convolution in "number grid cell". (2.5 is the standard value) 
_C.epn.deform_radius = 5.0 # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
_C.epn.KP_extent = 1.0 # Kernel point influence radius
_C.epn.KP_influence = 'linear' # Influence function when d < KP_extent. ('constant', 'linear', 'gaussian') When d > KP_extent, always zero
_C.epn.aggregation_mode = 'sum' # Aggregation function of KPConv in ('closest', 'sum') # Decide if you sum all kernel point influences, or if you only take the influence of the closest KP
_C.epn.fixed_kernel_points = 'verticals' # 'center' for KPConv, # Fixed points in the kernel : 'none', 'center' or 'verticals' # 'verticals' combined with equiv_mode_kp=True results in z-axis symmetric kernel
_C.epn.modulated = False # Use modulateion in deformable convolutions


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
