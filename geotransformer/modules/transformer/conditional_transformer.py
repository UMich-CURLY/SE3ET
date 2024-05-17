import torch.nn as nn
import torch

from geotransformer.modules.transformer.lrpe_transformer import LRPETransformerLayer
from geotransformer.modules.transformer.pe_transformer import PETransformerLayer
from geotransformer.modules.transformer.rpe_transformer import RPETransformerLayer
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer
from geotransformer.modules.transformer.output_layer import RotCompressOutput

from sklearn.metrics.pairwise import cosine_similarity
import geotransformer.modules.transformer.utils_epn.anchors as L
import numpy as np

def _check_block_type(block):
    # if block not in ['self', 'cross']:
    if 'self' not in block and 'cross' not in block:
        raise ValueError('Unsupported block type "{}".'.format(block))

def  _check_block_eq(block):
    list_eq = ['self_eq', 'cross_eq', 'cross_a_soft', 'cross_a_best', 'cross_r_soft', 'cross_r_best']
    return True if block in list_eq else False

def  _check_block_attn_mode(block):
    if '_eq' in block:
        return None
    else:
        list_mode = ['a_soft', 'a_best', 'r_soft', 'r_best']
        for mode in list_mode:
            if mode in block:
                return mode
        return None

class VanillaConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(VanillaConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class PEConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(PEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(PETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, embeddings1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


def _gather_impl(feat, idx, dense):
    if dense:
        return torch.gather(feat, 1, idx)
    else:
        return feat.flatten(0,1)[idx]

class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        return_attention_weights=False,
        anchor_matching=False,
        parallel=False,
        na=4,
        attn_r_positive='sq',
        attn_r_positive_rot_supervise='sigmoid',
        align_mode='0',
        alternative_impl=False,
        d_equiv_embed=0,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        self.align_mode = align_mode    # '0', '1', 'dual_early', 'dual_late'
        self.alternative_impl = alternative_impl
        self.d_equiv_embed = d_equiv_embed
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            equivariant = _check_block_eq(block)
            if 'self' in block:
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn, equivariant=equivariant, d_equiv_embed=d_equiv_embed))
            else:
                assert 'cross' in block, block
                attn_mode = _check_block_attn_mode(block)
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn, equivariant=equivariant, attn_mode=attn_mode, alternative_impl=alternative_impl, kanchor=na, attn_r_positive=attn_r_positive, attn_r_positive_rot_supervise=attn_r_positive_rot_supervise))
        self.layers = nn.ModuleList(layers)
        if 'cross_r_soft' in self.blocks or 'cross_r_best' in self.blocks:
            self.rotcompress = RotCompressOutput(d_model, dropout=dropout, activation_fn=activation_fn, na=na, dual_align=align_mode=='dual_early')
        self.return_attention_scores = return_attention_scores
        self.return_attention_weights = return_attention_weights
        self.anchor_matching = anchor_matching
        self.parallel = parallel
        # initialization of the anchors
        self.vertices, _, _, _, _ = L.get_octahedron_vertices()

    def eq2inv_normal(self, feats0, feats1, normal0, normal1):
        ### equivariant to invariant, banc->bnc
        # find nearest anchor from normals in batch
        similarities0 = cosine_similarity(normal0, self.vertices)
        anchor_idx0 = np.argmax(similarities0, axis=1) # p,1
        feats0_out = torch.empty_like(feats0[:,0,:,:])
        for i in range(feats0.shape[2]):
            feats0_out[:, i, :] = feats0[:, anchor_idx0[i], i, :]

        similarities1 = cosine_similarity(normal1, self.vertices)
        anchor_idx1 = np.argmax(similarities1, axis=1) # p,1
        feats1_out = torch.empty_like(feats1[:,0,:,:])
        for i in range(feats1.shape[2]):
            feats1_out[:, i, :] = feats1[:, anchor_idx1[i], i, :]

        return feats0_out, feats1_out
    
    def eq2inv_best(self, feats0, feats1, attn_w0, attn_w1, current_layer):
        ### permute
        assert not current_layer.attention.attention.attn_r_multihead, current_layer.attention.attention.attn_r_multihead
        if self.align_mode == '0':
            feats0_inv = feats0
            feats1_inv = _gather_impl(feats1, attn_w0, self.alternative_impl)
        elif self.align_mode == '1':
            feats0_inv = _gather_impl(feats0, attn_w1, self.alternative_impl)
            feats1_inv = feats1
        elif self.align_mode == 'dual_early':
            feats1_permute = _gather_impl(feats1, attn_w0, self.alternative_impl)   # banc -> banc
            feats0_permute = _gather_impl(feats0, attn_w1, self.alternative_impl)
            feats0_inv = torch.cat([feats0, feats0_permute], dim=1)  # b 2a n c
            feats1_inv = torch.cat([feats1_permute, feats1], dim=1)  # b 2a n c
        elif self.align_mode == 'dual_late':
            feats1_inv = _gather_impl(feats1, attn_w0, self.alternative_impl)   # banc -> banc
            feats0_inv = _gather_impl(feats0, attn_w1, self.alternative_impl)
        else:
            raise ValueError(f'self.align_mode {self.align_mode} not recognized')
        ### compress: equivariant to invariant
        feats0_inv = self.rotcompress(feats0_inv)
        feats1_inv = self.rotcompress(feats1_inv)
        if self.align_mode == 'dual_late':
            feats0 = self.rotcompress(feats0)
            feats1 = self.rotcompress(feats1)
            feats0 = feats0 + feats0_inv
            feats1 = feats1 + feats1_inv
        else:
            feats0 = feats0_inv
            feats1 = feats1_inv
        return feats0, feats1
    
    def eq2inv_soft(self, feats0, feats1, attn_w0, attn_w1, current_layer):
        ### permute
        # attn_w: brahnm, but only br(h) are effective
        assert not current_layer.attention.attention.attn_r_multihead, current_layer.attention.attention.attn_r_multihead
        attn_w0 = attn_w0[..., 0]   # brahnm -> brahn can be interpolated as branc
        attn_w1 = attn_w1[..., 0]   # brahnm -> brahn can be interpolated as branc
        if self.align_mode == '0':
            feats0_inv = feats0
            feats1_permute = feats1[:, current_layer.attention.attention.trace_idx_ori]  # branc
            feats1_inv = (feats1_permute * attn_w0).sum(1)  # banc
        elif self.align_mode == '1':
            feats0_permute = feats0[:, current_layer.attention.attention.trace_idx_ori]  # branc
            feats0_inv = (feats0_permute * attn_w1).sum(1)  # banc
            feats1_inv = feats1
        elif self.align_mode == 'dual_early':
            feats1_permute = feats1[:, current_layer.attention.attention.trace_idx_ori]  # branc
            feats0_permute = feats0[:, current_layer.attention.attention.trace_idx_ori]  # branc
            feats1_permute = (feats1_permute * attn_w0).sum(1)  # banc
            feats0_permute = (feats0_permute * attn_w1).sum(1)  # banc
            feats0_inv = torch.cat([feats0, feats0_permute], dim=1)  # b 2a n c
            feats1_inv = torch.cat([feats1_permute, feats1], dim=1)  # b 2a n c
        elif self.align_mode == 'dual_late':
            feats0_permute = feats0[:, current_layer.attention.attention.trace_idx_ori]  # branc
            feats0_inv = (feats0_permute * attn_w1).sum(1)  # banc
            feats1_permute = feats1[:, current_layer.attention.attention.trace_idx_ori]  # branc
            feats1_inv = (feats1_permute * attn_w0).sum(1)  # banc
        else:
            raise ValueError(f'self.align_mode {self.align_mode} not recognized')
            
        ### compress
        feats0_inv = self.rotcompress(feats0_inv)
        feats1_inv = self.rotcompress(feats1_inv)
        if self.align_mode == 'dual_late':
            feats0 = self.rotcompress(feats0)
            feats1 = self.rotcompress(feats1)
            feats0 = feats0 + feats0_inv
            feats1 = feats1 + feats1_inv
        else:
            feats0 = feats0_inv
            feats1 = feats1_inv
        return feats0, feats1

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None, 
                gt_indices=None, gt_overlap=None, equiv_embed0=None, equiv_embed1=None, 
                ref_normal=None, src_normal=None):
        attention_scores = []
        attn_matrix0 = None
        attn_matrix1 = None
        feats0_eq = None
        feats1_eq = None
        ref_feat_m = None
        src_feat_m = None
        for i, block in enumerate(self.blocks):
            if 'self' in block:
                if (feats0_eq is not None) and (feats1_eq is not None):
                    ### not the first self_eq block, use the equivariant features from the last cross block
                    feats0, scores0 = self.layers[i](feats0_eq, feats0_eq, embeddings0, memory_masks=masks0, equiv_states=equiv_embed0)
                    feats1, scores1 = self.layers[i](feats1_eq, feats1_eq, embeddings1, memory_masks=masks1, equiv_states=equiv_embed1)
                else:
                    ### the first self_eq block, use the features from the backbone
                    feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0, equiv_states=equiv_embed0)
                    feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1, equiv_states=equiv_embed1)
                if (block == 'self_eq') and (self.blocks[i+1] == 'cross'):
                    ### if the next block is cross, need to pool to invariant features
                    feats0_eq = feats0
                    feats1_eq = feats1
                    if (ref_normal is not None) and (src_normal is not None):
                        # use normal for pooling, banc->bnc 
                        feats0, feats1 = self.eq2inv_normal(feats0, feats1, ref_normal, src_normal)
                    else:
                        # max pool to invariant features
                        # feats0 = torch.mean(feats0_eq, dim=1, keepdim=False) # bahnc -> bhnc
                        # feats1 = torch.mean(feats1_eq, dim=1, keepdim=False) # bahnc -> bhnc
                        feats0 = torch.amax(feats0_eq, 1, keepdim=False) # bahnc -> bhnc
                        feats1 = torch.amax(feats1_eq, 1, keepdim=False) # bahnc -> bhnc
            else:
                assert 'cross' in block, block
                if self.parallel:
                    ### parallel=Ture doesn't converge when testing with 10 blocks, need to test again with new config
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1, gt_indices=gt_indices, gt_overlap=gt_overlap)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0, gt_indices=gt_indices, gt_overlap=gt_overlap)
                    feats0 = new_feats0
                    feats1 = new_feats1
                    ### TODO: add self_eq with cross mode for parallel
                else:
                    if (block == 'cross') and (i+1 < len(self.blocks)):  
                        if (self.blocks[i+1] == 'self_eq'):
                            ### current block is cross and next block is self_eq, need to obtain equivariant features
                            feats0_eq, scores0 = self.layers[i](feats0, feats1, feats1_eq, memory_masks=masks1, gt_indices=gt_indices, gt_overlap=gt_overlap)
                            # feats0 = torch.mean(feats0_eq, dim=1, keepdim=False) # bahnc -> bhnc
                            feats0 = torch.amax(feats0_eq, 1, keepdim=False) # bahnc -> bhnc
                            feats1_eq, scores1 = self.layers[i](feats1, feats0, feats0_eq, memory_masks=masks0, gt_indices=gt_indices, gt_overlap=gt_overlap)
                            # feats1 = torch.mean(feats1_eq, dim=1, keepdim=False) # bahnc -> bhnc
                            feats1 = torch.amax(feats1_eq, 1, keepdim=False) # bahnc -> bhnc
                        else:
                            feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1, gt_indices=gt_indices, gt_overlap=gt_overlap)
                            feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0, gt_indices=gt_indices, gt_overlap=gt_overlap)
                    elif (block == 'cross') and (i+1 == len(self.blocks)) and (self.blocks[i-1] == 'self_eq'):
                        ### the last cross block, we need to obtain both equivariant features and invariant features
                        feats0_eq, scores0 = self.layers[i](feats0, feats1, feats1_eq, memory_masks=masks1, gt_indices=gt_indices, gt_overlap=gt_overlap)
                        # feats0 = torch.mean(feats0_eq, dim=1, keepdim=False) # bahnc -> bhnc
                        feats0 = torch.amax(feats0_eq, 1, keepdim=False) # bahnc -> bhnc
                        feats1_eq, scores1 = self.layers[i](feats1, feats0, feats0_eq, memory_masks=masks0, gt_indices=gt_indices, gt_overlap=gt_overlap)
                        # feats1 = torch.mean(feats1_eq, dim=1, keepdim=False) # bahnc -> bhnc
                        feats1 = torch.amax(feats1_eq, 1, keepdim=False) # bahnc -> bhnc
                        ref_feat_m = feats0_eq
                        src_feat_m = feats1_eq
                    elif ('r_soft' in block) and (i+1 == len(self.blocks)) and self.return_attention_weights:
                        if (feats0_eq is not None) and (feats1_eq is not None):
                            ### the last cross block is equivariant cross_r and we are doing rotation supervision
                            ### so we need to obtain both equivariant features and invariant features
                            feats0_eq, scores0 = self.layers[i](feats0_eq, feats1_eq, memory_masks=masks1, gt_indices=gt_indices, gt_overlap=gt_overlap)
                            feats1_eq, scores1 = self.layers[i](feats1_eq, feats0_eq, memory_masks=masks0, gt_indices=gt_indices, gt_overlap=gt_overlap)
                            # feats0 = torch.mean(feats0_eq, dim=1, keepdim=False) # bahnc -> bhnc
                            # feats1 = torch.mean(feats1_eq, dim=1, keepdim=False) # bahnc -> bhnc
                            feats0 = torch.amax(feats0_eq, 1, keepdim=False) # bahnc -> bhnc
                            feats1 = torch.amax(feats1_eq, 1, keepdim=False) # bahnc -> bhnc
                            ref_feat_m = feats0_eq
                            src_feat_m = feats1_eq
                        else:
                            feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1, gt_indices=gt_indices, gt_overlap=gt_overlap)
                            feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0, gt_indices=gt_indices, gt_overlap=gt_overlap)
                    else:
                        ### normal cross block or the last cross block when no rotation supervision                       
                        feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1, gt_indices=gt_indices, gt_overlap=gt_overlap)
                        feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0, gt_indices=gt_indices, gt_overlap=gt_overlap)
                if 'r_soft' in block or 'r_best' in block:
                    if 'r_soft' in block:
                        # scores0, attn_w0, attn_matrix0, ref_feat_m = scores0
                        # scores1, attn_w1, attn_matrix1, src_feat_m = scores1
                        scores0, attn_w0, attn_matrix0, _ = scores0
                        scores1, attn_w1, attn_matrix1, _ = scores1
                        if (i+1 < len(self.blocks)):
                            ref_feat_m = feats0
                            src_feat_m = feats1
                    else:
                        scores0, attn_w0 = scores0  # , v_permute1
                        scores1, attn_w1 = scores1  # , v_permute0
                    if i+1 < len(self.blocks) and not _check_block_eq(self.blocks[i+1]):
                        ### if the next block is not equivariant, need to pool to invariant features
                        if 'r_best' in block:
                            feats0, feats1 = self.eq2inv_best(feats0, feats1, attn_w0, attn_w1, self.layers[i])
                        else:
                            # r_soft
                            if (ref_normal is not None) and (src_normal is not None):
                                # use normal for pooling
                                feats0, feats1 = self.eq2inv_normal(feats0, feats1, ref_normal, src_normal)
                            else:
                                feats0_eq = None
                                feats1_eq = None
                                feats0, feats1 = self.eq2inv_soft(feats0, feats1, attn_w0, attn_w1, self.layers[i])
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
            """
            # checking attention matrix after each layer
            if (feats0_eq is not None):
                print('layer', i, 'block name', block)
                # test similarity
                from einops import rearrange
                import torch.nn.functional as F
                feats0_norm = F.normalize(rearrange(feats0_eq, 'b a n c -> b a (n c)'), dim=-1)
                attention_matrix = torch.einsum('bac,bec->bae', feats0_norm, feats0_norm)
                attention_matrix = attention_matrix.mean(0) # bae
                print('transformer attention_matrix\n', attention_matrix)
            elif (block == 'self_eq') or ('r_soft' in block) or ('a_soft' in block):
                print('layer', i, 'block name', block)
                # test similarity
                print('feats0', feats0.shape)
                from einops import rearrange
                import torch.nn.functional as F
                feats0_norm = F.normalize(rearrange(feats0, 'b a n c -> b a (n c)'), dim=-1)
                attention_matrix = torch.einsum('bac,bec->bae', feats0_norm, feats0_norm)
                attention_matrix = attention_matrix.mean(0) # bae
                print('transformer attention_matrix\n', attention_matrix)
            """

        if self.return_attention_scores:
            return feats0, feats1, attention_scores#, v_permute0, v_permute1
        elif self.return_attention_weights or self.anchor_matching:
            return feats0, feats1, ref_feat_m, src_feat_m, attn_matrix0, attn_matrix1
        else:
            return feats0, feats1#, v_permute0, v_permute1


class LRPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        num_embeddings,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
    ):
        super(LRPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(
                    LRPETransformerLayer(
                        d_model, num_heads, num_embeddings, dropout=dropout, activation_fn=activation_fn
                    )
                )
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, emb_indices0, emb_indices1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, emb_indices0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, emb_indices1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1
