r"""Vanilla Transformer without positional embeddings.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from geotransformer.modules.layers import build_dropout_layer
from geotransformer.modules.transformer.output_layer import AttentionOutput
import geotransformer.modules.transformer.utils_epn.anchors as L
from geotransformer.modules.geotransformer.superpoint_matching import SuperPointMatching
from geotransformer.modules.geotransformer.superpoint_target import SuperPointTargetGenerator
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),'../e2pn','vgtk') )
import vgtk.so3conv as sptk
import vgtk.functional as fr

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(
        self, input_q, input_k, input_v, key_weights=None, key_masks=None, attention_factors=None, attention_masks=None, gt_indices=None, gt_overlap=None,
    ):
        """Vanilla Self-attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, N, C)
            input_k (Tensor): input tensor for key (B, M, C)
            input_v (Tensor): input tensor for value (B, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, H, N, M), attention scores before dropout
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        if input_v.ndim == 4:
            v = rearrange(self.proj_v(input_v), 'b a m (h c) -> b a h m c', h=self.num_heads)
        elif input_v.ndim == 3:
            v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        if attention_masks is not None:
            attention_scores = attention_scores.masked_fill(attention_masks, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        if input_v.ndim == 4:
            attention_scores = attention_scores.unsqueeze(1)
            hidden_states = torch.matmul(attention_scores, v)
            hidden_states = rearrange(hidden_states, 'b a h n c -> b a n (h c)')
        elif input_v.ndim == 3:
            hidden_states = torch.matmul(attention_scores, v)
            hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores

class MultiHeadAttentionEQ(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, attn_mode=None, alternative_impl=False, kanchor=4, attn_r_positive='sq', attn_r_positive_rot_supervise='sigmoid'):
        """
        The equivariant attention has four steps. 
            1. calculate the local attention matrix (per-point-pair, per-anchor-pair inner products);
            2. calculate the global rotation-wise (r_soft or r_best) or anchor-wise (a_soft or a_best) attention matrix, shared across the whole point cloud;
            3. use the global attention to weight (a_soft or r_soft) or select the best dim in (a_best or r_best) local attention;
            4. calculate the attended feature by multiplying the attention matrix with value feature. 

        attn_mode: when calculating global attention, how to attend query anchors to key anchors
            'a_soft': each anchor in query attends to each anchor in key;
            'a_best': each anchor in query finds the best-matchng anchor in key;
            'r_soft': anchors in key are permuted according to each rotation, each rotation is attended by query;
            'r_best': find the best-matching rotation (permutation of anchors) in key and attend by query;
            None: do not calculate global attention. Skip step 2 and 3 above. 
        attn_r_positive: when calculating global attention, how to make non-negative attention value before normalization, needed if attn_mode is soft
            'sq': square on the attention value;
            'abs': take absolute on the attention value;
            'relu': take relu on the attention value;
            'sigmoid': take sigmoid on the attention value;
            None: do not make the attention value non-negative.
        attn_r_summ: when calculating global attention, how to pool over all points
            'mean': average over all point pairs; 
            'double_norm': double normalization on rows and columns of point pair matrix, pick the topk matching pairs and take the average;
        attn_on_sub: if attn_mode == 'r_soft' or 'r_best', optional to use a subset of anchors to determine the best rotation or weight of each rotation
            True: only attend first two anchors of the query point cloud with key point cloud; 
            False: use all anchors of the query point cloud for attention.
        attn_r_multihead: use the same or different global attention for different attention heads.
            True: attention on different heads separately; 
            False: attention averaged over different heads.
        alternative_impl: different implementations of gathering which are equivalent
            True: use torch.gather, might consume more memory;
            False: use flat indexing.
        """
        super(MultiHeadAttentionEQ, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        # self.quotient_factor = quotient_factor

        self.attn_on_sub = False
        self.attn_r_summ = 'mean'
        self.attn_r_multihead = False
        self.attn_r_positive = attn_r_positive # 'sq', 'abs', 'relu', 'sigmoid', None
        self.attn_r_positive_rot_supervise = attn_r_positive_rot_supervise
        self.num_correspondences = 256
        self.dual_normalization = True
        self.num_targets = 128
        self.overlap_threshold = 0.1
        # self.attn_r_soft = False
        # self.attn_ra_soft = False
        self.attn_mode = attn_mode   # 'r_best' 'r_soft' 'a_best' 'a_soft'
        self.alternative_impl = alternative_impl
        self.kanchor = kanchor

        # # obselete (only used in cross_anchor_attn_ra)
        # self.attn_r_soft = False
        # self.attn_ra_soft = False

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

        if self.attn_r_positive == 'leakyrelu' or self.attn_r_positive_rot_supervise == 'leakyrelu':
            self.leakyrelu = nn.LeakyReLU(0.1)
        if self.attn_r_positive == 'softplus' or self.attn_r_positive_rot_supervise == 'softplus':
            self.softplus = nn.Softplus(beta=1.0)

        self.coarse_matching = SuperPointMatching(
            self.num_correspondences, self.dual_normalization
        )

        self.coarse_target = SuperPointTargetGenerator(
            self.num_targets, self.overlap_threshold
        )

        self.init_anchors()
    def init_anchors(self):
        if self.kanchor == 12:
            # anchors = L.get_anchorsV()
            vs, v_adjs, v_level2s, v_opps, vRs = L.get_icosahedron_vertices()
            self.adj0 = v_adjs[0,0]
            self.anchors = nn.Parameter(torch.tensor(vRs, dtype=torch.float32), requires_grad=False)  # 60*3*3
            trace_idx_ori, trace_idx_rot = L.get_relativeV_index()  # 60*12, 60*12 (ra)
            self.trace_idx_ori = nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.int64), requires_grad=False)
            self.trace_idx_rot = nn.Parameter(torch.tensor(trace_idx_rot, dtype=torch.int64), requires_grad=False)
            self.nr = trace_idx_ori.shape[0]
            self.na = trace_idx_ori.shape[1]
        elif self.kanchor == 6:
            vs, v_adjs, vRs, ecs, face_normals = L.get_octahedron_vertices()
            self.adj0 = v_adjs[0,0]
            self.anchors = nn.Parameter(torch.tensor(vRs, dtype=torch.float32), requires_grad=False)  # 12*3*3
            trace_idx_ori, trace_idx_rot = fr.get_relativeV_index(vRs, vs)
            self.trace_idx_ori = nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.int64), requires_grad=False)
            self.trace_idx_rot = nn.Parameter(torch.tensor(trace_idx_rot, dtype=torch.int64), requires_grad=False)
            self.nr = trace_idx_ori.shape[0]
            self.na = trace_idx_ori.shape[1]
        elif self.kanchor == 4:
            vs, v_adjs, vRs, ecs, face_normals = sptk.get_tetrahedron_vertices()
            self.adj0 = v_adjs[0,0]
            self.anchors = nn.Parameter(torch.tensor(vRs, dtype=torch.float32), requires_grad=False)  # 12*3*3
            trace_idx_ori, trace_idx_rot = fr.get_relativeV_index(vRs, vs)
            self.trace_idx_ori = nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.int64), requires_grad=False)
            self.trace_idx_rot = nn.Parameter(torch.tensor(trace_idx_rot, dtype=torch.int64), requires_grad=False)
            self.nr = trace_idx_ori.shape[0]
            self.na = trace_idx_ori.shape[1]
            """
            trace_idx_ori
            [[0 1 2 3]                                                                                            
            [0 2 3 1]                                                                                             
            [0 3 1 2]                                                                                             
            [1 2 0 3]
            [1 0 3 2]
            [1 3 2 0]
            [2 3 0 1]
            [2 0 1 3]
            [2 1 3 0]
            [3 1 0 2]
            [3 0 2 1]
            [3 2 1 0]]
            trace_idx_rot
            [[0 1 2 3]
            [0 3 1 2]
            [0 2 3 1]
            [2 0 1 3]
            [1 0 3 2]
            [3 0 2 1]
            [2 3 0 1]
            [1 2 0 3]
            [3 1 0 2]
            [2 1 3 0]
            [1 3 2 0]
            [3 2 1 0]]
            """
        elif self.kanchor == 3:            
            self.anchors = nn.Parameter(torch.tensor(L.get_anchors(self.kanchor), dtype=torch.float32), requires_grad=False)
            trace_idx_ori, trace_idx_rot = fr.get_relativeR_index(self.anchors)  
            trace_idx_rot = trace_idx_rot.transpose(0,1)
            # trace_idx_ori
            # [[0 1 2]
            # [1 2 0]
            # [2 0 1]]
            # trace_idx_rot
            # [[0 2 1]
            # [1 0 2]
            # [2 1 0]]
            self.trace_idx_ori = nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.int64), requires_grad=False)
            self.trace_idx_rot = nn.Parameter(torch.tensor(trace_idx_rot, dtype=torch.int64), requires_grad=False)
            self.nr = trace_idx_ori.shape[0]
            self.na = trace_idx_ori.shape[1]
        else:
            raise NotImplementedError(f'self.kanchor={self.kanchor} not implemented')
        return

    def cross_anchor_attn_aa(self, q, k, input_q, input_k, gt_indices=None, gt_overlap=None):
        '''q: bahnc, k: behmc -> 
        attention_scores: baehnm / brahnm / bahnm
        where: b: batch size; 
        a and e: number of anchors; 
        h: number of heads in the multi-head attention;
        n and m: number of points;
        return: local attention matrix and global attention matrix.
        '''
        b = q.shape[0]
        n = q.shape[-2]
        m = k.shape[-2]

        # print('q', q.amax(), q.amin(), q.mean())

        ####################
        ### normalization ##
        ####################
        # q_normalized = F.normalize(q, dim=-1) # normalize on the c dimension
        # k_normalized = F.normalize(k, dim=-1)
        # print('differences between q and k:', torch.norm(q - k))
        # print('self.attn_mode', self.attn_mode)
        # print('differences between nomalized q and k:', torch.norm(q_normalized - k_normalized))
        # print('q_normalized', q_normalized.amax(), q_normalized.amin(), q_normalized.mean())
        # print('k_normalized', k_normalized.amax(), k_normalized.amin(), k_normalized.mean())

        q_rearraged = rearrange(q, 'b a h n c -> b a h (n c)') # bahnc -> bah(nc) 
        q_normalized_nc = F.normalize(q_rearraged, dim=-1)
        q_normalized_nc = rearrange(q_normalized_nc, 'b a h (n c) -> b a h n c', n=n)

        k_rearraged = rearrange(k, 'b a h m c -> b a h (m c)')
        k_normalized_nc = F.normalize(k_rearraged, dim=-1)
        k_normalized_nc = rearrange(k_normalized_nc, 'b a h (m c) -> b a h m c', m=m)

        ####################
        ### local attention
        ####################
        if self.attn_on_sub:
            assert self.attn_mode == 'r_soft' or self.attn_mode == 'r_best', self.attn_mode
            attention_scores_ae = torch.einsum('bahnc,behmc->baehnm', q[:,[0,self.adj0]], k) \
                                    / self.d_model_per_head ** 0.5
        else:
            attention_scores_ae = torch.einsum('bahnc,behmc->baehnm', q, k) \
                                / self.d_model_per_head ** 0.5
            # attention_scores_ae_normalized = torch.einsum('bahnc,behmc->baehnm', q_normalized, k_normalized) \
            #                     / self.d_model_per_head ** 0.5
            if self.attn_mode == 'a_soft' or self.attn_mode == 'a_best':
                attention_scores_ae_rot_sup = torch.einsum('bahnc,behmc->baehnm', q_normalized_nc, k_normalized_nc) \
                                    / self.d_model_per_head ** 0.5
            elif self.attn_mode == 'r_soft' or self.attn_mode == 'r_best':
                
                attention_scores_ae_rot_sup = torch.einsum('bahnc,behmc->baehnm', q_normalized_nc, k_normalized_nc) \
                                    / self.d_model_per_head ** 0.5
                attention_scores_ae_rot_sup = rearrange(attention_scores_ae_rot_sup, 'b a e h n m -> b a e h (n m)') # baehnm -> baeh(nm)
                attention_scores_ae_rot_sup = torch.amax(attention_scores_ae_rot_sup, dim=-1, keepdim=False) # baeh(nm)-> baeh
                q_inv = torch.mean(q, dim=1, keepdim=False) # bahnc -> bhnc
                q_inv = rearrange(q_inv, 'b h n c -> b n (h c)')[0] # bahnc -> nc
                q_inv = F.normalize(q_inv, dim=-1)
                
                """
                # pool on anchor dimension, equivariant to invariant for point matching, bahnc -> bhnc
                q_inv = torch.mean(q, dim=1, keepdim=False) # bahnc -> bhnc
                k_inv = torch.mean(k, dim=1, keepdim=False) # bahnc -> bhnc
                # q_inv = torch.amax(q, dim=1, keepdim=False) # bahnc -> bhnc
                # k_inv = torch.amax(k, dim=1, keepdim=False) # bahnc -> bhnc
                # rearrange to match dimention requirement, bhnc -> nc
                q_inv = rearrange(q_inv, 'b h n c -> b n (h c)')[0] # bahnc -> nc
                k_inv = rearrange(k_inv, 'b h m c -> b m (h c)')[0] # bahmc -> mc
                # normalize on dimension c
                q_inv = F.normalize(q_inv, dim=-1)
                k_inv = F.normalize(k_inv, dim=-1)

                # TODO: merge the following two blocks
                # find point matching using features
                # print('====> coarse matching starts in transformer <====')
                ref_corr_indices_from_feat, src_corr_indices_from_feat, node_corr_scores = self.coarse_matching(
                        q_inv, k_inv, None, None
                    )
                # print('====> coarse matching ends in transformer <====')
                
                # check indices
                # print('corr_indices from feat\n', torch.stack((ref_corr_indices_from_feat[:10], src_corr_indices_from_feat[:10]), dim=1))
                    
                # matching points using indices
                q_matching_feat = q[:, :, :, ref_corr_indices_from_feat, :] # bahnc -> bahn'c
                k_matching_feat = k[:, :, :, src_corr_indices_from_feat, :] # behmc -> behn'c, find the best matching point
                # normalize over nc
                temp_q_matching_feat = F.normalize(rearrange(q_matching_feat, 'b a h n c -> b a h (n c)'), dim=-1)
                q_matching_feat = rearrange(temp_q_matching_feat, 'b a h (n c) -> b a h n c', n=ref_corr_indices_from_feat.shape[0])
                temp_k_matching_feat = F.normalize(rearrange(k_matching_feat, 'b a h m c -> b a h (m c)'), dim=-1)
                k_matching_feat = rearrange(temp_k_matching_feat, 'b a h (m c) -> b a h m c', m=src_corr_indices_from_feat.shape[0])                
                # calculate attention matrix
                attention_scores_ae_rot_sup = torch.einsum('bahnc,behnc->baeh', q_matching_feat, k_matching_feat)
                if not self.attn_r_multihead:
                    attention_scores_ae_rot_sup_temp = attention_scores_ae_rot_sup.mean(3) # baenm
                    # print('attention matrix from feature matching\n', attention_scores_ae_rot_sup_temp)

                # during training, use ground truth indices for point matching
                if (gt_indices is not None) and (gt_overlap is not None):
                    # Random select ground truth node correspondences during training
                    ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                        gt_indices, gt_overlap
                    )
                    # matching points using indices
                    q_matching = q[:, :, :, ref_node_corr_indices, :] # bahnc -> bahn'c
                    k_matching = k[:, :, :, src_node_corr_indices, :] # behmc -> behn'c, find the best matching point
                    # normalize over nc
                    temp_q_matching = F.normalize(rearrange(q_matching, 'b a h n c -> b a h (n c)'), dim=-1)
                    q_matching = rearrange(temp_q_matching, 'b a h (n c) -> b a h n c', n=ref_node_corr_indices.shape[0])
                    temp_k_matching = F.normalize(rearrange(k_matching, 'b a h m c -> b a h (m c)'), dim=-1)
                    k_matching = rearrange(temp_k_matching, 'b a h (m c) -> b a h m c', m=src_node_corr_indices.shape[0])
                    # check indices
                    # print('corr_indices', len(ref_node_corr_indices), len(src_node_corr_indices))
                    # print('corr_indices\n', torch.stack((ref_node_corr_indices[:10], src_node_corr_indices[:10]), dim=1))
                    # calculate attention matrix
                    attention_scores_ae_rot_sup = torch.einsum('bahnc,behnc->baeh', q_matching, k_matching)
                """
            else:
                raise NotImplementedError(f'self.attn_mode={self.attn_mode} not implemented')
            
        # print('local attention_scores_ae', attention_scores_ae.shape)
        
        if self.attn_mode is None:
            return attention_scores_ae, None, None

        attention_scores_ae_raw = attention_scores_ae   # local attention
        # attention_scores_ae = torch.sigmoid(attention_scores_ae)   # 0-1
        # print('local', attention_scores_ae_rot_sup.amax(), attention_scores_ae_rot_sup.amin(), attention_scores_ae_rot_sup.mean())


        ####################
        ### global attention
        ####################
        if not self.attn_r_multihead:
            ### average over attention heads
            attention_scores_ae = attention_scores_ae.mean(3) # baenm
            # attention_scores_ae_normalized = attention_scores_ae_normalized.mean(3) # baenm
            attention_scores_ae_rot_sup = attention_scores_ae_rot_sup.mean(3) # baenm


        ## make the attention value non-negative for rotation/anchor normalization
        if self.attn_r_positive == 'sq':
            attention_scores_ae = attention_scores_ae**2
        elif self.attn_r_positive == 'abs':
            attention_scores_ae = torch.abs(attention_scores_ae)
        elif self.attn_r_positive == 'relu':
            attention_scores_ae = F.relu(attention_scores_ae)
        elif self.attn_r_positive == 'sigmoid':
            attention_scores_ae = F.sigmoid(attention_scores_ae)
        elif self.attn_r_positive == 'leadkyrelu':
            attention_scores_ae = self.leakyrelu(attention_scores_ae)
        elif self.attn_r_positive == 'softplus':
            attention_scores_ae = self.softplus(attention_scores_ae)


        if self.attn_r_positive_rot_supervise is None:
            pass
        elif self.attn_r_positive_rot_supervise == 'sq':
            attention_scores_ae_rot_sup = attention_scores_ae_rot_sup**2
        elif self.attn_r_positive_rot_supervise == 'abs':
            attention_scores_ae_rot_sup = torch.abs(attention_scores_ae_rot_sup)
        elif self.attn_r_positive_rot_supervise == 'relu':
            attention_scores_ae_rot_sup = F.relu(attention_scores_ae_rot_sup)
        elif self.attn_r_positive_rot_supervise == 'sigmoid':
            attention_scores_ae_rot_sup = F.sigmoid(attention_scores_ae_rot_sup)
        elif self.attn_r_positive_rot_supervise == 'leadkyrelu':
            attention_scores_ae_rot_sup = self.leakyrelu(attention_scores_ae_rot_sup)
        elif self.attn_r_positive_rot_supervise == 'softplus':
            attention_scores_ae_rot_sup = self.softplus(attention_scores_ae_rot_sup)
        elif self.attn_r_positive_rot_supervise == 'minus':
            attention_scores_ae_rot_sup = (attention_scores_ae_rot_sup + 1) / 2
        
        
        # print('norm', attention_scores_ae_rot_sup.amax(), attention_scores_ae_rot_sup.amin(), attention_scores_ae_rot_sup.mean())

        # print('global attention_scores_ae', attention_scores_ae.shape)

        ### pool over all points
        if self.attn_r_summ == 'mean':
            attn_ae = attention_scores_ae.mean([-2,-1])   # bae(h)
            # attn_ae_normalized = attention_scores_ae_normalized.mean([-2,-1])
            if self.attn_mode == 'a_soft' or self.attn_mode == 'a_best':
                attn_ae_rot_sup = attention_scores_ae_rot_sup.mean([-2,-1])
            elif self.attn_mode == 'r_soft' or self.attn_mode == 'r_best':
                attn_ae_rot_sup = attention_scores_ae_rot_sup
        elif self.attn_r_summ == 'double_norm':
            ref_matching_scores = attention_scores_ae / attention_scores_ae.sum(dim=-1, keepdim=True)
            src_matching_scores = attention_scores_ae / attention_scores_ae.sum(dim=-2, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
            num_correspondences = min(self.num_correspondences, matching_scores.numel())
            corr_scores, corr_indices = matching_scores.flatten(-2).topk(k=num_correspondences, largest=True)    # bae(h)k
            attn_ae = corr_scores.mean(-1)          # bae(h)

            # ref_matching_scores_normalized = attn_ae_normalized / attn_ae_normalized.sum(dim=-1, keepdim=True)
            # src_matching_scores_normalized = attn_ae_normalized / attn_ae_normalized.sum(dim=-2, keepdim=True)
            # matching_scores_normalized = ref_matching_scores_normalized * src_matching_scores_normalized
            # num_correspondences_normalized = min(self.num_correspondences, matching_scores_normalized.numel())
            # corr_scores_normalized, corr_indices_normalized = matching_scores_normalized.flatten(-2).topk(k=num_correspondences, largest=True)    # bae(h)k
            # attn_ae_normalized = corr_scores_normalized.mean(-1)          # bae(h)

            ref_matching_scores_rot_sup = attention_scores_ae_rot_sup / attention_scores_ae_rot_sup.sum(dim=-1, keepdim=True)
            src_matching_scores_rot_sup = attention_scores_ae_rot_sup / attention_scores_ae_rot_sup.sum(dim=-2, keepdim=True)
            matching_scores_rot_sup = ref_matching_scores_rot_sup * src_matching_scores_rot_sup
            num_correspondences_rot_sup = min(self.num_correspondences, matching_scores_rot_sup.numel())
            corr_scores_rot_sup, corr_indices_rot_sup = matching_scores_rot_sup.flatten(-2).topk(k=num_correspondences, largest=True)    # bae(h)k
            attn_ae_rot_sup = corr_scores_rot_sup.mean(-1)          # bae(h)
        else:
            raise NotImplementedError(f'attn_r_summ ={self.attn_r_summ} not recognized')
        
        # print('attn_ae_rot_sup', attn_ae_rot_sup)

        # attn_ae_rot_sup = F.sigmoid(attn_ae_rot_sup)
        # print('sigmoid', attn_ae_rot_sup)

        
        # print('self.attn_mode', self.attn_mode, 'attn_ae_normalized', attn_ae_normalized)

        # print('attn_ae', attn_ae.shape, '\n', attn_ae)
        
        if self.attn_mode == 'a_soft':
            ### calculate the global anchor attention weight
            assert self.attn_r_positive is not None, "Normalization should be conducted on non-negative weights."
            ### normalize over anchors in key 
            attn_ae = attn_ae / attn_ae.sum(2, keepdim=True)    # bae(h)
            attn_ae = attn_ae[..., None, None]
            if not self.attn_r_multihead:
                attn_ae = attn_ae.unsqueeze(3)    # baehnm
            attention_scores = attention_scores_ae_raw   # baehnm

            return attention_scores, attn_ae
            ### attention_scores: local attention matrix; attn_ae: global anchor attention matrix
        elif self.attn_mode == 'a_best':
            ### pick the optimal anchor in key for each anchor in query
            attn_a_max, attn_a_max_idx = attn_ae.max(dim=2, keepdim=True) # ba1(h)
            if self.alternative_impl:
                if self.attn_r_multihead:
                    attn_a_max_idx = attn_a_max_idx[..., None, None].expand(-1, -1, -1, -1, n, m)  # baehnm
                else:
                    attn_a_max_idx = attn_a_max_idx[..., None, None, None].expand(-1, -1, -1, self.num_heads, n, m)  # baehnm
                attention_scores = torch.gather(attention_scores_ae_raw, 2, attn_a_max_idx) # ba1hnm
                attention_scores = attention_scores.squeeze(2)  # bahnm
            else:
                if self.attn_r_multihead:
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1,1) # b,1,1,1
                    lin_idx_a = torch.arange(self.na, device=q.device).reshape(1,-1,1,1) # b,1,1,1
                    attn_ae_idx = lin_idx_b * self.na + lin_idx_a
                    attn_ae_idx = attn_ae_idx * self.na + attn_a_max_idx
                    lin_idx_h = torch.arange(self.num_heads, device=q.device).reshape(1,1,1,-1) # b,1,1,1
                    attn_ae_idx = attn_ae_idx * self.num_heads + lin_idx_h  # bae(=1)h
                    attention_scores = attention_scores_ae_raw.flatten(0,3)[attn_ae_idx].squeeze(2) # baehnm -> bahnm
                else:
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1) # b,1,1
                    lin_idx_a = torch.arange(self.na, device=q.device).reshape(1,-1,1) # b,1,1
                    attn_ae_idx = lin_idx_b * self.na + lin_idx_a
                    attn_ae_idx = attn_ae_idx * self.na + attn_a_max_idx  # bae(=1)
                    attention_scores = attention_scores_ae_raw.flatten(0,2)[attn_ae_idx].squeeze(2) # baehnm -> bahnm

            return attention_scores, attn_a_max_idx
            ### attention_scores: local attention matrix; attn_ae: global best matching anchor indices
        elif self.attn_mode == 'r_soft': 
            ### calculate the global rotation attention weight
            assert self.attn_r_positive is not None, "Normalization should be conducted on non-negative weights."
            if self.alternative_impl:
                if self.attn_on_sub:
                    trace_idx_ori = self.trace_idx_ori[:, [0,self.adj0]]  # 60*2
                else:
                    trace_idx_ori = self.trace_idx_ori
                attn_are = attn_ae[:,:,trace_idx_ori] # bae(h) -> bare(h)
                if self.attn_r_multihead:
                    attn_r = torch.einsum('barah -> brh', attn_are)
                    ### normalize over rotations in key 
                    attn_r = attn_r / attn_r.sum(1, keepdim=True)
                    attn_r = attn_r.unsqueeze(2)[..., None, None]   # brahnm
                else:
                    attn_r = torch.einsum('bara -> br', attn_are)
                    ### normalize over rotations in key 
                    attn_r = attn_r / attn_r.sum(1, keepdim=True)
                    attn_r = attn_r[..., None, None, None, None]    # brahnm

                # print('attn_r', attn_r.squeeze())
                k_permute = k[:, self.trace_idx_ori]    # behmc -> brahmc
                attention_scores = torch.einsum('bahnc, brahmc -> brahnm', q, k_permute) \
                                    / self.d_model_per_head ** 0.5
                attn_r_new = attention_scores.mean((2,3,4,5))   # br
                # attn_r_new = torch.sigmoid(attention_scores).mean((2,3,4,5))   # br
                attn_r_new = attn_r_new / attn_r_new.sum(1, keepdim=True)
                # print('attn_r_new', attn_r_new)
                # print('---------------')

            else:
                if self.attn_on_sub:
                    trace_idx_ori2 = self.trace_idx_ori[:, [0,self.adj0]]   # 60*2
                    trace_idx_ori2 = trace_idx_ori2.transpose(0,1)[None]    # 1*2*60 bar
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1)        # bar(b11)
                    lin_idx_a = torch.arange(2, device=q.device).reshape(1,-1,1)  # bar(1a1)
                    idx = lin_idx_b * 2 + lin_idx_a
                    idx = idx * self.na + trace_idx_ori2
                    attn_ar = attn_ae.flatten(0,2)[idx] # bae(h) -> bar(h)
                    attn_r = attn_ar.mean(1)   # br(h)
                    attn_r = attn_r / attn_r.sum(1, keepdim=True)
                    attn_r = attn_r.unsqueeze(2)[..., None, None]   # bra(h)nm
                    if not self.attn_r_multihead:
                        attn_r = attn_r.unsqueeze(3)    # brahnm

                    attention_scores_ae = torch.einsum('bahnc,behmc->baehnm', q, k) \
                                        / self.d_model_per_head ** 0.5
                    trace_idx_ori = self.trace_idx_ori.transpose(0,1)[None]  # 1*12*60 bar
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1)        # bar(b11)
                    lin_idx_a = torch.arange(self.na, device=q.device).reshape(1,-1,1)  # bar(1a1)
                    idx = lin_idx_b * self.na + lin_idx_a
                    idx = idx * self.na + trace_idx_ori     # bar
                    attention_scores = attention_scores_ae.flatten(0,2)[idx]    # barhnm
                    attention_scores = attention_scores.transpose(1,2)          # brahnm
                else:
                    # attn_are = attn_ae[:,:,self.trace_idx_ori] # bare(h)
                    trace_idx_ori = self.trace_idx_ori.transpose(0,1)[None]  # 1*12*60 bar
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1)        # bar(b11)
                    lin_idx_a = torch.arange(self.na, device=q.device).reshape(1,-1,1)  # bar(1a1)
                    idx = lin_idx_b * self.na + lin_idx_a
                    idx = idx * self.na + trace_idx_ori     # bar
                    attn_ar = attn_ae.flatten(0,2)[idx] # bae(h) -> bar(h)
                    attn_r = attn_ar.mean(1)   # br(h)
                    attn_r = attn_r / attn_r.sum(1, keepdim=True)
                    attn_r = attn_r.unsqueeze(2)[..., None, None]   # bra(h)nm
                    if not self.attn_r_multihead:
                        attn_r = attn_r.unsqueeze(3)    # brahnm
                    
                    attention_scores = attention_scores_ae_raw.flatten(0,2)[idx]   # baehnm -> barhnm
                    attention_scores = attention_scores.transpose(1,2)             # brahnm

            return attention_scores, [attn_r, attn_ae_rot_sup, q_inv]
            ### attention_scores: local attention matrix; attn_ae: global rotation attention matrix
        elif self.attn_mode == 'r_best':
            ### pick the optimal rotation in key and permute key anchors accordingly
            if self.alternative_impl:
                if self.attn_on_sub:
                    trace_idx_ori = self.trace_idx_ori[:, [0,self.adj0]]  # 60*2
                else:
                    trace_idx_ori = self.trace_idx_ori
                attn_are = attn_ae[:,:,trace_idx_ori] # bae(h) -> bare(h)
                if self.attn_r_multihead:
                    attn_r = torch.einsum('barah -> brh', attn_are)
                else:
                    attn_r = torch.einsum('bara -> br', attn_are)
                attn_r_max, attn_r_max_idx = attn_r.max(dim=1) # b(h)
                trace_idx_ori = self.trace_idx_ori[attn_r_max_idx]  # ra -> b(h)a
                if self.attn_r_multihead:
                    trace_idx_ori = trace_idx_ori.transpose(1,2)[..., None, None].expand_as(k)    # bahmc
                else:
                    trace_idx_ori = trace_idx_ori[..., None, None, None].expand_as(k)        # bahmc
                k_permute = torch.gather(k, 1, trace_idx_ori)
                attention_scores = torch.einsum('bahnc,bahmc->bahnm', q, k_permute) \
                                        / self.d_model_per_head ** 0.5
            else:
                if self.attn_on_sub:
                    trace_idx_ori2 = self.trace_idx_ori[:, [0,self.adj0]]  # 60*2
                    # attn_are = attn_ae[:,:,trace_idx_ori2] # bae(h) -> bare(h)
                    trace_idx_ori2 = trace_idx_ori2.transpose(0,1)[None]  # 1*2*60 bar
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1)        # bar(b11)
                    lin_idx_a = torch.arange(2, device=q.device).reshape(1,-1,1)  # bar(1a1)
                    idx = lin_idx_b * 2 + lin_idx_a
                    idx = idx * self.na + trace_idx_ori2
                    attn_ar = attn_ae.flatten(0,2)[idx] # bae(h) -> bar(h)
                    attn_r = attn_ar.mean(1)   # br(h)
                else:
                    trace_idx_ori = self.trace_idx_ori.transpose(0,1)[None]  # 1*12*60 bar
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1)        # bar(b11)
                    lin_idx_a = torch.arange(self.na, device=q.device).reshape(1,-1,1)  # bar(1a1)
                    idx = lin_idx_b * self.na + lin_idx_a
                    idx = idx * self.na + trace_idx_ori     # bar
                    attn_ar = attn_ae.flatten(0,2)[idx] # bae(h) -> bar(h)
                    attn_r = attn_ar.mean(1)   # br(h)
                # attn_r = attn_are.mean((1,3))   # br(h)
                attn_r_max, attn_r_max_idx = attn_r.max(dim=1) # b(h)
                trace_idx_ori = self.trace_idx_ori[attn_r_max_idx]  # ra -> b(h)a
                if self.attn_r_multihead:
                    trace_idx_ori = trace_idx_ori.transpose(1,2)    #bah
                    lin_idx_b = torch.arange(b).reshape(-1,1,1) # b,1,1
                    trace_idx_ori = lin_idx_b * self.na + trace_idx_ori   # b,a,h
                    lin_idx_h = torch.arange(self.num_heads).reshape(1,1,-1)
                    trace_idx_ori = trace_idx_ori * self.num_heads + lin_idx_h
                    k_permute = k.flatten(0,2)[trace_idx_ori] #bahmc
                    attention_scores = torch.einsum('bahnc,bahmc->bahnm', q, k_permute) \
                                            / self.d_model_per_head ** 0.5
                else:
                    lin_idx_b = torch.arange(b, device=trace_idx_ori.device).reshape(-1,1) # b,1
                    trace_idx_ori = lin_idx_b * self.na + trace_idx_ori   # b,a
                    k_permute = k.flatten(0,1)[trace_idx_ori] #bahmc
                    attention_scores = torch.einsum('bahnc,bahmc->bahnm', q, k_permute) \
                                            / self.d_model_per_head ** 0.5

            return attention_scores, trace_idx_ori
            ### attention_scores: local attention matrix; trace_idx_ori: anchor permutation unber the global best matching rotation
        else:
            raise NotImplementedError(f"self.attn_mode={self.attn_mode} not recognized")

    def cross_anchor_attn_ra(self, q, k):
        '''q: bahnc, k: bahmc -> bahnm'''
        ### try all permutations corresponding to rotations, and calculate attention

        if self.attn_on_sub:
            trace_idx_ori2 = self.trace_idx_ori[:, [0,self.adj0]]  # 60*2
            k2_permute = k[:,trace_idx_ori2]   # b r a(=2) h m c
            attention_scores_ra = torch.einsum('bahnc,brahmc->brahnm', q[:,[0,self.adj0]], k2_permute) \
                                    / self.d_model_per_head ** 0.5
        else:
            k_permute = k[:,self.trace_idx_ori]   # b r a h m c
            attention_scores_ra = torch.einsum('bahnc,brahmc->brahnm', q, k_permute) \
                                    / self.d_model_per_head ** 0.5

        attention_scores_ra_raw = attention_scores_ra

        attention_scores_ra = torch.sigmoid(attention_scores_ra)   # 0-1

        ### pool spatial-wise
        if not self.attn_r_multihead:
            attention_scores_ra = attention_scores_ra.mean(3) # branm
        if self.attn_r_positive == 'sq':
            attention_scores_ra = attention_scores_ra**2    # make sure not negative before this
        elif self.attn_r_positive == 'abs':
            attention_scores_ra = torch.abs(attention_scores_ra)    # make sure not negative before this
        elif self.attn_r_positive == 'relu':
            attention_scores_ra = F.relu(attention_scores_ra)
        elif self.attn_r_positive == 'sigmoid':
            attention_scores_ra = F.sigmoid(attention_scores_ra)

        if self.attn_r_summ == 'mean':
            attn_ra = attention_scores_ra.mean([-2,-1])   # bra(h)
        elif self.attn_r_summ == 'double_norm':
            ref_matching_scores = attention_scores_ra / attention_scores_ra.sum(dim=-1, keepdim=True)
            src_matching_scores = attention_scores_ra / attention_scores_ra.sum(dim=-2, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
            num_correspondences = min(self.num_correspondences, matching_scores.numel())
            corr_scores, corr_indices = matching_scores.flatten(-2).topk(k=num_correspondences, largest=True)    # bra(h)k
            attn_ra = corr_scores.mean(-1)          # bra(h)
        else:
            raise NotImplementedError(f'attn_r_summ ={self.attn_r_summ} not recognized')

        ### pick the optimal rotation and permute k accordingly
        if self.attn_r_soft:
            attn_ra = attn_ra / attn_ra.sum(1, keepdim=True)     #bra(h)
            if self.attn_on_sub:
                pass
            else:
                if not self.attn_ra_soft:
                    attn_ra = attn_ra.mean(2, keepdim=True)   # br1(h)
                if not self.attn_r_multihead:
                    attn_ra_weight = attn_ra_weight[..., None]  #brah

                attn_ra_weight = attn_ra[..., None,None]    #brah -> brahnm
                attention_scores = (attention_scores_ra_raw * attn_ra_weight).sum(1)    # bahnm
                    
        else:
            attn_r = attn_ra.mean(2)   # br(h)
            attn_r_max, attn_r_max_idx = attn_r.max(dim=1) # b(h)
            ### calculate final attention
            if self.attn_on_sub:
                ### need to recalculate attention (initial attn is only a subset)
                ### first permute k
                trace_idx_ori = self.trace_idx_ori[attn_r_max_idx]  # ra -> b(h)a
                b = q.shape[0]
                n = q.shape[3]
                m = k.shape[3]
                if self.attn_r_multihead:
                    trace_idx_ori = trace_idx_ori.transpose(1,2)    #bah
                    lin_idx = torch.arange(b).reshape(-1,1,1) # b,1,1
                    trace_idx_ori = lin_idx * self.na + trace_idx_ori   # b,a,h

                    lin_idx_h = torch.arange(self.num_heads).reshape(1,1,-1)
                    trace_idx_ori = trace_idx_ori * self.num_heads + lin_idx_h

                    k_permute = k.flatten(0,2)[trace_idx_ori] #bahmc
                else:
                    lin_idx = torch.arange(b).reshape(-1,1) # b,1
                    trace_idx_ori = lin_idx * self.na + trace_idx_ori   # b,a

                    k_permute = k.flatten(0,1)[trace_idx_ori] #bahmc

                ### then calc attn scores
                attention_scores = torch.einsum('bahnc,bahmc->bahnm', q, k_permute) \
                                        / self.d_model_per_head ** 0.5

            else:
                ### gather from initial attention
                if self.attn_r_multihead:
                    attn_r_max_idx = attn_r_max_idx.unsqueeze(1)    # br(=1)h
                    lin_idx_b = torch.arange(b).reshape(-1,1,1) # b,1,1
                    attn_r_max_idx = lin_idx_b * self.nr + attn_r_max_idx   # b,r,h
                    
                    lin_idx_h = torch.arange(self.num_heads).reshape(1,1,-1)
                    attn_r_max_idx = attn_r_max_idx * self.num_heads + lin_idx_h    # b,r,h

                    attention_scores = attention_scores_ra_raw.transpose(2,3).flatten(0, 2 \
                        )[attn_r_max_idx].reshape(b,1,self.num_heads,self.na,n,m).transpose(2,3).squeeze(1)
                        # bahnm
                else:
                    attn_r_max_idx = attn_r_max_idx.unsqueeze(1)    # br(=1)
                    lin_idx_b = torch.arange(b).reshape(-1,1) # b,1
                    attn_r_max_idx = lin_idx_b * self.nr + attn_r_max_idx   # b,r
                    attention_scores = attention_scores_ra_raw.flatten(0,1)[attn_r_max_idx].reshape(b,self.na,self.num_heads,n,m)
                    # bahnm

        return attention_scores

    def forward(
        self, input_q, input_k, input_v, key_weights=None, key_masks=None, attention_factors=None, attention_masks=None, gt_indices=None, gt_overlap=None
    ):
        """Vanilla Self-attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, A, N, C)
            input_k (Tensor): input tensor for key (B, A, M, C)
            input_v (Tensor): input tensor for value (B, A, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, A, N, C)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, A, A, H, N, M), local attention scores
                'attn_w' ('attn_idx'): global rotation or anchor attention scores (or best indices)
        """

        # print('differences between input q and k:', torch.norm(input_q - input_k))

        q = rearrange(self.proj_q(input_q), 'b a n (h c) -> b a h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b a m (h c) -> b a h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b a m (h c) -> b a h m c', h=self.num_heads)

        # share linear layer, not recomended, just testing rotation supervision
        # k = rearrange(self.proj_q(input_k), 'b a m (h c) -> b a h m c', h=self.num_heads)
        # v = rearrange(self.proj_q(input_v), 'b a m (h c) -> b a h m c', h=self.num_heads)
        q_rearrange = rearrange(input_q, 'b a n (h c) -> b a h n c', h=self.num_heads)
        k_rearrange = rearrange(input_k, 'b a m (h c) -> b a h m c', h=self.num_heads)


        attention_scores, attn_w = self.cross_anchor_attn_aa(q, k, q_rearrange, k_rearrange, gt_indices, gt_overlap)     
        ### local and global attention matrix, of the same shape: baehnm / brahnm / bahnm
        
        if self.attn_mode in ['a_best', 'r_best']:
            attn_idx = attn_w

        if self.attn_mode in ['a_best', 'r_best']:
            if attention_factors is not None:
                attention_scores = attention_factors.unsqueeze(1).unsqueeze(1) * attention_scores
            if key_weights is not None:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            if key_masks is not None:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1).unsqueeze(1), float('-inf'))
            if attention_masks is not None:
                attention_scores = attention_scores.masked_fill(attention_masks, float('-inf'))
        else:   # None, 'a_soft', 'r_soft'
            if attention_factors is not None:
                attention_scores = attention_factors.unsqueeze(1).unsqueeze(1).unsqueeze(1) * attention_scores
            if key_weights is not None:
                attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            if key_masks is not None:
                attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1), float('-inf'))
            if attention_masks is not None:
                attention_scores = attention_scores.masked_fill(attention_masks, float('-inf'))
               

        ### normalize the local attention over the points in key
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        if self.attn_mode == 'a_soft':
            ### attention_scores: baehnm, attn_w: baehnm, v: behmc
            attention_scores = attention_scores * attn_w
            hidden_states = torch.einsum('baehnm,behmc->bahnc', attention_scores, v)
        elif self.attn_mode == 'a_best':
            if self.alternative_impl:
                attn_idx = attn_idx[:,:,0,:,0,:, None].expand_as(v)   # bahmc
                v_permute = torch.gather(v, 1, attn_idx)
            else:
                ### attention_scores: bahnm, attn_idx: bae(=1)(h), v: behmc
                attn_idx = attn_idx.squeeze(2)  # ba(h)
                if self.attn_r_multihead:
                    b = attention_scores.shape[0]
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1,1) # b,1,1
                    attn_idx = lin_idx_b * self.na + attn_idx
                    lin_idx_h = torch.arange(self.num_heads, device=q.device).reshape(1,1,-1) # b,1,1
                    attn_idx = attn_idx * self.num_heads + lin_idx_h  # bah
                    v_permute = v.flatten(0,2)[attn_idx] # bahmc
                else:
                    b = attention_scores.shape[0]
                    lin_idx_b = torch.arange(b, device=q.device).reshape(-1,1) # b,1
                    attn_idx = lin_idx_b * self.na + attn_idx
                    v_permute = v.flatten(0,1)[attn_idx] # bahmc
            hidden_states = torch.einsum('bahnm,bahmc->bahnc', attention_scores, v_permute)
        elif self.attn_mode == 'r_soft':
            ### brahnm, brahnm, behmc
            attn_w, attn_matrix, feat_m = attn_w
            attention_scores = attention_scores * attn_w
            v_permute = v[:, self.trace_idx_ori]    # brahmc
            # print('v_permute_0000', v_permute[..., 0,0,0,0])
            hidden_states = torch.einsum('brahnm,brahmc->bahnc', attention_scores, v_permute)
        elif self.attn_mode == 'r_best':
            if self.alternative_impl:
                ### attn_idx: bahmc
                v_permute = torch.gather(v, 1, attn_idx)
            else:
                ### attention_scores: bahnm, attn_idx: ba(h), v: behmc
                if self.attn_r_multihead:
                    v_permute = v.flatten(0,2)[attn_idx]    # bahmc
                else:
                    v_permute = v.flatten(0,1)[attn_idx]    # bahmc
            hidden_states = torch.einsum('bahnm,bahmc->bahnc', attention_scores, v_permute)
        else:
            assert self.attn_mode is None, self.attn_mode
            hidden_states = torch.einsum('baehnm,behmc->bahnc', attention_scores, v)

        # hidden_states = torch.matmul(attention_scores, v)   # bahnm, bahmc -> bahnc

        hidden_states = rearrange(hidden_states, 'b a h n c -> b a n (h c)')

        if self.attn_mode in ['a_best', 'r_best']:
            return hidden_states, [attention_scores, attn_idx]
        elif self.attn_mode in ['r_soft']:
            return hidden_states, [attention_scores, attn_w, attn_matrix, feat_m]
        else:
            return hidden_states, [attention_scores, attn_w]

class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, equivariant=False, attn_mode=None, alternative_impl=False, kanchor=4, attn_r_positive='sq', attn_r_positive_rot_supervise='sigmoid'):
        super(AttentionLayer, self).__init__()
        self.equivariant = equivariant
        if self.equivariant:
            self.attention = MultiHeadAttentionEQ(d_model, num_heads, dropout=dropout, attn_mode=attn_mode, alternative_impl=alternative_impl, kanchor=kanchor, attn_r_positive=attn_r_positive, attn_r_positive_rot_supervise=attn_r_positive_rot_supervise)
        else:
            self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        value_states=None,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
        gt_indices=None,
        gt_overlap=None,
    ):
        if value_states is None:
            value_states = memory_states
        hidden_states, attention_scores = self.attention(
            input_states, # q
            memory_states, # k
            value_states, # v
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
            gt_indices=gt_indices,
            gt_overlap=gt_overlap,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU', equivariant=False, attn_mode=None, alternative_impl=False, kanchor=4, attn_r_positive='sq', attn_r_positive_rot_supervise='sigmoid'):
        super(TransformerLayer, self).__init__()
        self.equivariant = equivariant
        self.attention = AttentionLayer(d_model, num_heads, dropout=dropout, equivariant=equivariant, attn_mode=attn_mode, alternative_impl=alternative_impl, kanchor=kanchor, attn_r_positive=attn_r_positive, attn_r_positive_rot_supervise=attn_r_positive_rot_supervise)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        value_states=None,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
        gt_indices=None,
        gt_overlap=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            value_states=value_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
            gt_indices=gt_indices,
            gt_overlap=gt_overlap,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = AttentionLayer(d_model, num_heads, dropout=dropout)
        self.cross_attention = AttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(self, input_states, memory_states, input_masks=None, memory_masks=None):
        hidden_states, attention_scores = self.self_attention(input_states, input_states, memory_masks=input_masks)
        hidden_states, attention_scores = self.cross_attention(hidden_states, memory_states, memory_masks=memory_masks)
        output_states = self.output(hidden_states)
        return output_states, attention_scores


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=None, activation_fn='ReLU'):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)

    def forward(self, feats, weights=None, masks=None, attention_factors=None, attention_masks=None):
        r"""Transformer Encoder forward.

        Args:
            feats (Tensor): (B, N, C)
            weights (Tensor=None): (B, N)
            masks (BoolTensor=None): True if ignored (B, N)
            attention_factors (Tensor=None): (B, N, N)
            attention_masks (BoolTensor=None): (B, N, N)

        Returns:
            feats (Tensor): (B, N, C)
        """
        for i in range(self.num_layers):
            feats, _ = self.layers[i](
                feats,
                feats,
                memory_weights=weights,
                memory_masks=masks,
                attention_factors=attention_factors,
                attention_masks=attention_masks,
            )
        return feats


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=None, activation_fn='ReLU'):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerDecoderLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)

    def forward(self, q_feats, s_feats):
        r"""Transformer Decoder forward.

        Args:
            q_feats (Tensor): (B, N, C)
            s_feats (Tensor): (B, M, C)

        Returns:
            q_feats (Tensor): (B, N, C)
        """
        for i in range(self.num_layers):
            q_feats, _ = self.layers[i](q_feats, s_feats)
        return q_feats
