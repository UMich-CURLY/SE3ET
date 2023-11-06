import torch
import torch.nn as nn

from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.modules.ops.pairwise_distance import pairwise_distance

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../e2pn','vgtk') )
import vgtk.so3conv as sptk
import vgtk.functional as fr

class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        # ref_feats = output_dict['ref_feats_c'] # bnc
        # src_feats = output_dict['src_feats_c'] # bmc
        ref_feats = output_dict['ref_feats_m'] # banc
        src_feats = output_dict['src_feats_m'] # bamc
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)

        # anchor_matching = torch.zeros_like(ref_feats)
        # pos_masks = torch.gt(overlaps, self.positive_overlap)
        # neg_masks = torch.eq(overlaps, 0)

        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss
    

class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class RotationMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(RotationMatchingLoss, self).__init__()
        """
        For kanchor = 4, the number of positive and negative examples are 1 to 3, thus pos_weight = 3
        mode 0: attn_w0 P0 = P1
        mode 1: attn_w1 P1 = P0
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.na = cfg.epn.kanchor
        if self.na == 4:
            # define the anchors
            vs, v_adjs, vRs, ecs, face_normals = sptk.get_tetrahedron_vertices()
            self.adj0 = v_adjs[0,0]
            self.anchors = nn.Parameter(torch.tensor(vRs, dtype=torch.float32), requires_grad=False)  # 12*3*3
            self.trace_idx_ori, _ = fr.get_relativeV_index(vRs, vs) # 12*4, 12*4
            # trace_idx_ori
            # [[0 1 2 3]
            # [0 2 3 1]
            # [0 3 1 2]
            # [1 2 0 3]
            # [1 0 3 2]
            # [1 3 2 0]
            # [2 3 0 1]
            # [2 0 1 3]
            # [2 1 3 0]
            # [3 1 0 2]
            # [3 0 2 1]
            # [3 2 1 0]]
        elif self.na == 6:
            vs, v_adjs, vRs, ecs, face_normals = sptk.get_octahedron_vertices()
            self.adj0 = v_adjs[0,0]
            self.anchors = nn.Parameter(torch.tensor(vRs, dtype=torch.float32), requires_grad=False)  # 12*3*3
            self.trace_idx_ori, _ = fr.get_relativeV_index(vRs, vs)
        else:
            raise NotImplementedError(f"kanchor={self.na} is not implemented in the RotationMatchingLoss()")
        pos_weight = 3 * torch.ones((self.na, self.na)).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, output_dict, data_dict):
        """
        targets are either 0 or 1
        """
        """
        # load attention weights from output_dict
        attn_matrix0 = output_dict['attn_matrix0'].squeeze() # 4*4
        print('attn_matrix0\n', attn_matrix0.shape, '\n', attn_matrix0)
        attn_matrix1 = output_dict['attn_matrix1'].squeeze() # 4*4
        # print('attn_matrix1\n', attn_matrix1.shape, '\n', attn_matrix1)

        # get ground truth rotation matrix from data_dict
        # ref_points = src_points @ rotation.T + translation
        gt_T0 = data_dict['transform'] # (4, 4)
        gt_R0 = gt_T0[:3, :3] # (3, 3) 
        # print('gt_R0\n', gt_R0)

        # Find the nearest anchor from the ground truth rotation
        # T = R_target * anchors[label]
        R0_target, R0_label = fr.label_relative_rotation_simple(self.anchors, gt_R0)
        v0 = torch.Tensor(self.trace_idx_ori[R0_label]).long()
        index0 = v0.reshape((-1, 1)).to(self.device)
        # print('index0', index0)

        # transpose the rotation matrix for supervising attn_matrix1
        gt_R1 = gt_R0.T # (3, 3) 
        # print('gt_R1\n', gt_R1)
        R1_target, R1_label = fr.label_relative_rotation_simple(self.anchors, gt_R1)
        v1 = torch.Tensor(self.trace_idx_ori[R1_label]).long()
        index1 = v1.reshape((-1, 1)).to(self.device)
        # print('index1', index1)

        # contruct ground true label matrix
        src = torch.ones((4, 1)).to(self.device)
        target0 = torch.zeros(4, 4, dtype=attn_matrix0.dtype, device=attn_matrix0.device).scatter_(1, index0, src)
        target1 = torch.zeros(4, 4, dtype=attn_matrix1.dtype, device=attn_matrix1.device).scatter_(1, index1, src)

        if self.attn_r_positive_rot_supervise == 'softplus':
            target0 = self.softplus(target0)
            target1 = self.softplus(target1)

        print('target0\n', target0)
        # print('target1\n', target1)
        
        # attn_w0 P0 = P1, attn_w1 P1 = P0
        loss0 = self.criterion(attn_matrix0, target0)
        loss1 = self.criterion(attn_matrix1, target1)
        """

        # load attention weights from output_dict
        attn_matrix = output_dict['rot_sup_matrix'].squeeze() # 4*4
        print('attn_matrix\n', attn_matrix.shape, '\n', attn_matrix)

        # get ground truth rotation matrix from data_dict
        gt_T0 = data_dict['transform'] # (4, 4)
        gt_R0 = gt_T0[:3, :3] # (3, 3) 

        # Find the nearest anchor from the ground truth rotation
        R_target, R0_label = fr.label_relative_rotation_simple(self.anchors, gt_R0)
        v = torch.Tensor(self.trace_idx_ori[R0_label]).long()
        index = v.reshape((-1, 1)).to(self.device)

        # contruct ground true label matrix
        src = torch.ones((4, 1)).to(self.device)
        target = torch.zeros(4, 4, dtype=attn_matrix.dtype, device=attn_matrix.device).scatter_(1, index, src)
        print('target\n', target)

        loss = self.criterion(attn_matrix, target)

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)      
        self.fine_loss = FineMatchingLoss(cfg)
        self.rotation_loss = RotationMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss
        self.weight_rotation_loss = cfg.loss.weight_rotation_loss
        self.supervise_rotation = cfg.geotransformer.supervise_rotation
        self.anchor_matching = cfg.geotransformer.anchor_matching

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        if self.supervise_rotation:
            rotation_loss = self.rotation_loss(output_dict, data_dict)
            loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss + self.weight_rotation_loss * rotation_loss

            return {
                'loss': loss,
                'c_loss': coarse_loss,
                'f_loss': fine_loss,
                'r_loss': rotation_loss,
            }
        
        else:
            loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

            return {
                'loss': loss,
                'c_loss': coarse_loss,
                'f_loss': fine_loss,
            }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']

        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }
