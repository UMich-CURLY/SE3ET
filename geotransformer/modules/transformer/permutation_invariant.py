import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../e2pn','vgtk') )
import vgtk.so3conv as sptk
import vgtk.functional as fr

class PermutationInvariantLayer(nn.Module):
    def __init__(self, na, d_model):
        super(PermutationInvariantLayer, self).__init__()
        self.fc1 = nn.Linear(na*d_model, na*d_model)
        self.batch_norm = nn.BatchNorm1d(na*d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(na*d_model, d_model)

        self.na = na
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
            raise NotImplementedError(f"kanchor={self.na} is not implemented in the PermutationInvariantLayer()")
        

    def forward(self, ref_feats_m, src_feats_m, gt_T0):
        # get ground truth rotation matrix from data_dict
        gt_R0 = gt_T0[:3, :3] # (3, 3)

        # Find the nearest anchor from the ground truth rotation
        R_target, R0_label = fr.label_relative_rotation_simple(self.anchors, gt_R0)
        v = torch.Tensor(self.trace_idx_ori[R0_label]).long() 

        # get the matching anchor features
        src_feats_matching = src_feats_m[:, v] # benc -> banc (matching anchor with the feat0)

        # MLP for reducing the size of the features      
        ref_feats_m = rearrange(ref_feats_m, 'b a n c -> b n (a c)')
        src_feats_matching = rearrange(src_feats_matching, 'b a n c -> b n (a c)')

        # ref_feats = self.fc1(ref_feats_m)
        # ref_feats = self.dropout(ref_feats)
        # ref_feats = torch.permute(ref_feats, (0, 2, 1))
        # ref_feats = self.batch_norm(ref_feats)
        # ref_feats = torch.permute(ref_feats, (0, 2, 1))
        # ref_feats = self.relu(ref_feats)
        ref_feats = ref_feats_m
        ref_feats = self.fc2(ref_feats)
        
        # src_feats = self.fc1(src_feats_matching)
        # src_feats = self.dropout(src_feats)
        # src_feats = torch.permute(src_feats, (0, 2, 1))
        # src_feats = self.batch_norm(src_feats)
        # src_feats = torch.permute(src_feats, (0, 2, 1))
        # src_feats = self.relu(src_feats)
        src_feats = src_feats_matching
        src_feats = self.fc2(src_feats)
        
        return ref_feats, src_feats