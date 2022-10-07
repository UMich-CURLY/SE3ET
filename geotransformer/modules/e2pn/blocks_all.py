from geotransformer.modules.e2pn.blocks import *
from geotransformer.modules.e2pn.blocks_epn import *

def block_decider(block_name,
                  radius,
                  in_dim,
                  out_dim,
                  layer_ind,
                  config):

    if block_name == 'unary':
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)

    elif block_name in ['simple',
                        'simple_deformable',
                        'simple_invariant',
                        'simple_equivariant',
                        'simple_strided',
                        'simple_deformable_strided',
                        'simple_invariant_strided',
                        'simple_equivariant_strided']:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name in ['resnetb',
                        'resnetb_invariant',
                        'resnetb_equivariant',
                        'resnetb_deformable',
                        'resnetb_strided',
                        'resnetb_deformable_strided',
                        'resnetb_equivariant_strided',
                        'resnetb_invariant_strided']:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name == 'unary_epn':
        return UnaryBlockEPN(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)

    elif block_name == 'simple_epn':
        return SimpleBlockEPN(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name in ['resnetb_epn', 
                        'resnetb_strided_epn']:
        return ResnetBottleneckBlockEPN(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name == 'inv_epn':
        return InvOutBlockEPN(block_name, in_dim, config)
    
    elif block_name == 'lift_epn':
        return LiftBlockEPN(block_name, in_dim, config)

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)
