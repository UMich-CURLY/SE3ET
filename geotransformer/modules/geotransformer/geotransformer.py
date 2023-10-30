import numpy as np
import torch
import torch.nn as nn
import e3nn.o3 as o3

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../e2pn','vgtk') )
import vgtk.so3conv as sptk
# import vgtk.functional as fr
# import geotransformer.modules.transformer.utils_epn.anchors as L

def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1

class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max', kanchor=1, n_level_equiv=0):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.n_level_equiv = n_level_equiv
        self.kanchor = kanchor
        if self.n_level_equiv > 0 and self.kanchor > 1:
            self.init_anchors()

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    def init_anchors(self):
        if self.kanchor == 12:
            vRs = sptk.get_anchorsV12()  # 12(in 60)*3*3
        elif self.kanchor == 4:
            vRs = sptk.get_anchorsV12(tetra=True)  # 4(in 12)*3*3
        elif self.kanchor == 6:
            vRs = sptk.get_anchorsV24()  # 6(in 24)*3*3
        else:
            raise NotImplementedError(f'self.kanchor={self.kanchor} not implemented')

        anchors = torch.tensor(vRs, dtype=torch.float32)
        ### list of length n_level_equiv, each is a tensor n_anchors*(2l+1)*(2l+1)
        self.anchors_wignerD = nn.ParameterList(
                            [nn.Parameter(o3.Irrep(l,1).D_from_matrix(anchors.transpose(1,2)), requires_grad=False) 
                            for l in range(self.n_level_equiv)])
        return

    @torch.no_grad()
    def get_equiv_embedding(self, points):
        ### BNM -> BANMD
        diff = points[:,:,None] - points[:,None]    # BN13-B1M3 = BNM3
        # all_degrees = list(range(2 * self.n_level_equiv + 1))
        all_degrees = list(range(self.n_level_equiv))
        sh = o3.spherical_harmonics(all_degrees, diff, normalize=True)  # BNMD
        ### list of length n_level_equiv， each BNM(2l+1)
        sh = torch.split(sh, [degree_to_dim(d) for d in all_degrees], dim=-1)
        sh_at_anchors = torch.cat([torch.einsum('acd,bnmd->banmc', self.anchors_wignerD[l], sh[l]) for l in all_degrees], -1)
        return sh_at_anchors

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            # a_embeddings = a_embeddings.max(dim=3)[0]
            a_embeddings = a_embeddings.amax(dim=3)
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        if self.n_level_equiv > 0:
            eq_embeddings = self.get_equiv_embedding(points)
            return embeddings, eq_embeddings
        else:
            return embeddings


class GeometricTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        supervise_rotation=False,
        anchor_matching=False,
        reduction_a='max',
        na=None,
        attn_r_positive='sq',
        attn_r_positive_rot_supervise='sigmoid',
        align_mode='0',
        alternative_impl=False,
        n_level_equiv=0,
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()
        self.n_level_equiv = n_level_equiv
        self.d_equiv_embed = (np.arange(self.n_level_equiv)*2+1).sum() # 0 if n_level_equiv==0 because it is empty
        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a, kanchor=na, n_level_equiv=n_level_equiv)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.na = na
        self.supervise_rotation = supervise_rotation
        self.anchor_matching = anchor_matching
        if self.na is None:
            # transformer in GeoTransformer
            self.transformer = RPEConditionalTransformer(
                blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
            )
        else:
            if self.supervise_rotation:
                # transformer that handle equivariant features and return attention weight for rotation supervision
                self.transformer = RPEConditionalTransformer(
                    blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn,
                    return_attention_weights=True,
                    na=na,
                    attn_r_positive=attn_r_positive,
                    attn_r_positive_rot_supervise=attn_r_positive_rot_supervise,
                    align_mode=align_mode,
                    alternative_impl=alternative_impl,
                    d_equiv_embed=self.d_equiv_embed,
                )
            elif self.anchor_matching:
                # transformer that handle equivariant features and return attention weight for rotation supervision
                self.transformer = RPEConditionalTransformer(
                    blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn,
                    anchor_matching=True,
                    na=na,
                    attn_r_positive=attn_r_positive,
                    attn_r_positive_rot_supervise=attn_r_positive_rot_supervise,
                    align_mode=align_mode,
                    alternative_impl=alternative_impl,
                    d_equiv_embed=self.d_equiv_embed,
                )
            else:
                # transformer that handle equivariant features
                self.transformer = RPEConditionalTransformer(
                    blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn,
                    na=na,
                    attn_r_positive=attn_r_positive,
                    attn_r_positive_rot_supervise=attn_r_positive_rot_supervise,
                    align_mode=align_mode,
                    alternative_impl=alternative_impl,
                    d_equiv_embed=self.d_equiv_embed,
                )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
        gt_indices=None,
        gt_overlap=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, [A], C)
            src_feats (Tensor): (B, M, [A], C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_feats_m = None
        src_feats_m = None
        attn_matrix0 = None
        ttn_matrix1 = None        

        if self.n_level_equiv == 0:
            ref_embeddings = self.embedding(ref_points)
            src_embeddings = self.embedding(src_points)
            ref_eq_embeddings = None
            src_eq_embeddings = None
        else:
            ref_embeddings, ref_eq_embeddings = self.embedding(ref_points)
            src_embeddings, src_eq_embeddings = self.embedding(src_points)

        if self.na is None:
            ref_feats = self.in_proj(ref_feats)
            src_feats = self.in_proj(src_feats)

            ref_feats, src_feats = self.transformer(
                ref_feats,
                src_feats,
                ref_embeddings,
                src_embeddings,
                masks0=ref_masks,
                masks1=src_masks,
                gt_indices=gt_indices,
                gt_overlap=gt_overlap,
            )

            ref_feats = self.out_proj(ref_feats)
            src_feats = self.out_proj(src_feats)
        else:
            # Use equivairant features from E2PN and geometric embeddings
            ref_feats = torch.transpose(ref_feats, 1, 2) # B, N, A, C -> B, A, N, C
            src_feats = torch.transpose(src_feats, 1, 2)
            ref_feats = self.in_proj(ref_feats)
            src_feats = self.in_proj(src_feats)

            if self.supervise_rotation or self.anchor_matching:
                ref_feats, src_feats, ref_feats_m, src_feats_m, attn_matrix0, attn_matrix1 = self.transformer(
                    ref_feats,
                    src_feats,
                    ref_embeddings,
                    src_embeddings,
                    masks0=ref_masks,
                    masks1=src_masks,
                    gt_indices=gt_indices,
                    gt_overlap=gt_overlap,
                    equiv_embed0=ref_eq_embeddings,
                    equiv_embed1=src_eq_embeddings,
                )
            else:
                ref_feats, src_feats = self.transformer(
                    ref_feats,
                    src_feats,
                    ref_embeddings,
                    src_embeddings,
                    masks0=ref_masks,
                    masks1=src_masks,
                    gt_indices=gt_indices,
                    gt_overlap=gt_overlap,
                    equiv_embed0=ref_eq_embeddings,
                    equiv_embed1=src_eq_embeddings,
                )

            ref_feats = self.out_proj(ref_feats) # B, N, C
            src_feats = self.out_proj(src_feats)
        
        return ref_feats, src_feats, ref_feats_m, src_feats_m, attn_matrix0, attn_matrix1
