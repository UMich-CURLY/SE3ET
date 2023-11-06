import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
from geotransformer.modules.transformer.rotation_supervision import RotationAttentionLayer
from geotransformer.modules.transformer.permutation_invariant import PermutationInvariantLayer
from backbone import E2PN
from einops import rearrange

class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.save_backbone_feature = cfg.geotransformer.save_backbone_feature
        self.anchor_matching = cfg.geotransformer.anchor_matching        

        self.backbone = E2PN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
            cfg.epn,
        )
        
        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            supervise_rotation=cfg.geotransformer.supervise_rotation,
            anchor_matching=cfg.geotransformer.anchor_matching,
            reduction_a=cfg.geotransformer.reduction_a,
            na=cfg.epn.kanchor,
            attn_r_positive=cfg.geotransformer.attn_r_positive,
            attn_r_positive_rot_supervise=cfg.geotransformer.attn_r_positive_rot_supervise,
            align_mode=cfg.geotransformer.align_mode,
            alternative_impl=cfg.geotransformer.alternative_impl,
            n_level_equiv=cfg.geotransformer.n_level_equiv,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)
        if self.save_backbone_feature:
            self.rotation_supervision = RotationAttentionLayer(cfg.geotransformer.input_dim, cfg.geotransformer.num_heads)
        else:
            self.rotation_supervision = RotationAttentionLayer(cfg.geotransformer.output_dim, cfg.geotransformer.num_heads)

        self.permutation_invariant = PermutationInvariantLayer(cfg.epn.kanchor, cfg.geotransformer.output_dim)

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c] # N, A, C=1024
        src_feats_c = feats_c[ref_length_c:]
        
        
        if self.save_backbone_feature and self.transformer.supervise_rotation:
            ref_feats_m = torch.permute(ref_feats_c.unsqueeze(0), (0, 2, 1, 3)) # B, A, N, C=1024
            src_feats_m = torch.permute(src_feats_c.unsqueeze(0), (0, 2, 1, 3)) # B, A, N, C=1024
            ref_feats_c, src_feats_c, _, _, attn_matrix0, attn_matrix1 = self.transformer(
                ref_points_c.unsqueeze(0),
                src_points_c.unsqueeze(0),
                ref_feats_c.unsqueeze(0),
                src_feats_c.unsqueeze(0),
            ) # B, N/M, C=256
        else:
            ref_feats_c, src_feats_c, ref_feats_m, src_feats_m, attn_matrix0, attn_matrix1 = self.transformer(
                ref_points_c.unsqueeze(0),
                src_points_c.unsqueeze(0),
                ref_feats_c.unsqueeze(0),
                src_feats_c.unsqueeze(0),
            ) # B, N/M, C=256
        
        output_dict['ref_feats_m'] = ref_feats_m
        output_dict['src_feats_m'] = src_feats_m
        output_dict['attn_matrix0'] = attn_matrix0
        output_dict['attn_matrix1'] = attn_matrix1

        if self.anchor_matching:
            # permutation invariant layer, make src_feat_m, src_feat_m invariant for anchor matching
            ref_feats_m, src_feats_m, ref_feats_c, src_feats_c = self.permutation_invariant(ref_feats_m, src_feats_m, data_dict['transform'])
            output_dict['ref_feats_m'] = ref_feats_m
            output_dict['src_feats_m'] = src_feats_m

        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f] # N, A, C=256
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        
        # 10. Inference Rotation
        with torch.no_grad():
            print('ref_feats_m', ref_feats_m.shape)
            ref_matching = ref_feats_m[:, :, ref_node_corr_indices, :] # banc -> ban'c
            src_matching = src_feats_m[:, :, src_node_corr_indices, :] # bemc -> ben'c, find the best matching point
            print('ref_matching', ref_matching.shape)

            # normalize over nc
            temp_ref_matching = F.normalize(rearrange(ref_matching, 'b a n c -> b a (n c)'), dim=-1)
            print('temp_ref_matching', temp_ref_matching.shape, 'n', ref_node_corr_indices.shape[0])
            ref_matching = rearrange(temp_ref_matching, 'b a (n c) -> b a n c', c=ref_matching.shape[-1])
            temp_src_matching = F.normalize(rearrange(src_matching, 'b a m c -> b a (m c)'), dim=-1)
            print('temp_src_matching', temp_src_matching.shape, 'm', src_node_corr_indices.shape[0])
            src_matching = rearrange(temp_src_matching, 'b a (m c) -> b a m c', c=src_matching.shape[-1])

            # calculate attention matrix
            attention_scores_ae_rot_sup = torch.einsum('bahnc,behnc->baeh', ref_matching, src_matching)
            attention_scores_ae_rot_sup = attention_scores_ae_rot_sup.mean(3) # bae
            attention_scores_ae_rot_sup = (attention_scores_ae_rot_sup + 1) / 2
            rot_sup_attn_matrix = self.rotation_supervision(ref_feats_m, src_feats_m, ref_node_corr_indices, src_node_corr_indices)
            output_dict['rot_sup_matrix'] = rot_sup_attn_matrix
            print('rot_sup_attn_matrix\n', rot_sup_attn_matrix)
            print('transform\n', data_dict['transform'])

        
        torch.cuda.empty_cache()

        return output_dict


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()