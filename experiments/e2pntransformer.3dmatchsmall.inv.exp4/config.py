import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)
ensure_dir(_C.registration_dir)

# data
_C.data = edict()
_C.data.dataset_root = osp.join(_C.root_dir, 'data', '3DMatch')

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
_C.train.point_limit = 5000 #10000 # random select if number of points is larger than point_limit
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.005
_C.train.augmentation_rotation = 1.0

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.point_limit = 5000 #30000 #None

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 0.2
_C.eval.rre_threshold = 15.0
_C.eval.rte_threshold = 0.3

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 12
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 40
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.init_voxel_size = 0.025
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 512


# epn
_C.epn = edict()
# ======== SO(2) ======== 
### Valid parameters: kanchor < 10, quotient_factor >= 1, k = 15
# _C.epn.kanchor = 3 # number of rotation anchors A; 12: E2PN, 60 & lower than 10: EPN
# _C.epn.quotient_factor = 1 # > 1 enables quotient features. 2 results in C6/C2. 
# _C.epn.num_kernel_points = 15 # Number of kernel points
# ======== E2PN ======== 
### Valid parameter set 1: kanchor = 12, quotient_factor  = 5, k = 13
# _C.epn.kanchor = 12 # number of rotation anchors A; 12: E2PN, 60 & lower than 10: EPN
# _C.epn.quotient_factor = 5 # > 1 enables quotient features. 2 results in C6/C2. 
# _C.epn.num_kernel_points = 13 # Number of kernel points
# _C.num_points = 4096
# ======== E2PN ======== 
### Valid parameter set 1: kanchor = 4, quotient_factor  = 3, k = 15
_C.epn.kanchor = 4 # number of rotation anchors A; 12: E2PN, 60 & lower than 10: EPN
_C.epn.quotient_factor = 3 # > 1 enables quotient features. 2 results in C6/C2. 
_C.epn.num_kernel_points = 15 # Number of kernel points
# _C.num_points = 4096
# ======== EPN ======== 
### Valid parameter set 2: kanchor = 60, quotient_factor  = 1, k = 13
# _C.epn.kanchor = 60 # number of rotation anchors A; 12: E2PN, 60 & lower than 10: EPN
# _C.epn.quotient_factor = 1 # > 1 enables quotient features. 2 results in C6/C2. 
# _C.epn.num_kernel_points = 13 # Number of kernel points
# _C.num_points = 2048
# ========
_C.epn.num_points = _C.train.point_limit
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

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05
_C.model.num_points_in_patch = 64
_C.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 512 #1024
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
# _C.geotransformer.blocks = ['self_eq', 'cross_a_best', 'self_eq', 'cross_r_best', 'self', 'cross']
# _C.geotransformer.blocks = ['self_eq', 'cross_a_soft', 'self_eq', 'cross_r_soft', 'self', 'cross']
 #['cross_r_soft', 'self_eq', 'cross_eq', 'cross_a_soft', 'cross_a_best', 'cross_r_best']
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.1
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.05

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


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
