"""
Script to test that Encoder kernel points and rotation anchors have the equivariance property. 
"""

import torch
import torch.nn as nn
import numpy as np

from geotransformer.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer
import geotransformer.modules.transformer.utils_epn.anchors as L
from geotransformer.modules.kpconv.kernel_points import load_kernels

class PseudoEncoder(nn.Module):
    '''Not an actual encoder, just initializing the kernel points and rotation anchors. '''
    def __init__(self, kanchor=12, quotient_factor=5, k=13) -> None:
        super().__init__()
        self.kanchor = kanchor
        self.quotient_factor = quotient_factor
        self.radius = 1
        self.K = k  # 13 for SO(3), 15 for SO(2)
        self.epn_kernel = False
        self.equiv_mode_kp = True
        self.ignore_steer_constraint = False
        self.non_sep_conv = True
        self.rot_by_permute = True
        self.gather_by_idxing = True

        self.kernel_points = self.init_KP()
        self.anchors = self.init_anchors()
        self.init_permute_idxs_kpts()
        self.init_permute_idxs_rots()

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        if self.kanchor < 10:
            K_points_numpy = load_kernels(self.radius,
                                    self.K,
                                    dimension=3,
                                    fixed='verticals',
                                    equiv_mode=self.equiv_mode_kp,
                                    )
        else:
            assert self.K == 13, self.K
            vs, v_adjs, v_level2s, v_opps, vRs = L.get_icosahedron_vertices()
            K_points_numpy = vs * 0.66 * self.radius
            K_points_numpy = np.concatenate([K_points_numpy, np.array([[0,0,0]], dtype=K_points_numpy.dtype)], axis=0)    # 13*3
    
        # ### can print the kernel points coordinates
        # print('K_points_numpy', K_points_numpy)
        # nk*3
        return nn.Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def init_anchors(self):
        # get so3 anchors (60x3x3 rotation matrices)
        if self.quotient_factor == 1:
            if self.kanchor < 10:
                ### EPN mode for SO(2)
                anchors = L.get_anchors(self.kanchor)
            else:
                ### EPN mode for SO(3)
                assert self.kanchor == 60, self.kanchor
                anchors = L.get_anchorsV()
        else:
            if self.kanchor < 10:
                ### E2PN mode for SO(2)
                anchors = L.get_anchors(self.kanchor * self.quotient_factor)[:self.kanchor]
                quotient_anchors = L.get_anchors(self.quotient_factor)
                self.quotient_anchors = nn.Parameter(torch.tensor(quotient_anchors, dtype=torch.float32),
                            requires_grad=False)
            else:
                ### E2PN mode for SO(3)
                assert self.kanchor == 12, self.kanchor
                assert self.quotient_factor == 5, self.quotient_factor
                anchors = L.get_anchorsV().reshape(12, 5, 3, 3)[:, 0]
                quotient_anchors = L.get_anchors(self.quotient_factor)
                self.quotient_anchors = nn.Parameter(torch.tensor(quotient_anchors, dtype=torch.float32),
                            requires_grad=False)

        return nn.Parameter(torch.tensor(anchors, dtype=torch.float32),
                         requires_grad=False)
    
    def init_permute_idxs_kpts(self):
        # self.anchors # a*3*3
        # self.kernel_points  # k*3
        rot_kps = torch.einsum('aij,kj->aki', self.anchors, self.kernel_points)
        kps_diff = rot_kps.unsqueeze(1) - self.kernel_points.reshape(1,self.K, 1, 3) # a,1,k2,3 - 1,k1,1,3
        kps_diff_norm = torch.norm(kps_diff, 2, -1) # a,k1,k2
        kres_ori, kidx_ori = torch.min(kps_diff_norm, 1)    # a, k2
        kres_rot, kidx_rot = torch.min(kps_diff_norm, 2)    # a, k1
        if self.quotient_factor > 1 and not self.ignore_steer_constraint:
            quotient_kps = torch.einsum('aij,kj->aki', self.quotient_anchors, self.kernel_points)
            kps_diffq = quotient_kps.unsqueeze(1) - self.kernel_points.reshape(1,self.K, 1, 3)
            kps_diff_normq = torch.norm(kps_diffq, 2, -1) # a,k1,k2
            kres_oriq, kidx_oriq = torch.min(kps_diff_normq, 1)    # a, k2
            exist_id = set()
            id_from_real = dict()
            id_to_real = torch.empty(self.K, dtype=torch.int64)
            real_i = 0
            for i in range(self.K):
                if i not in exist_id:
                    id_from_real[real_i] = set(kidx_oriq[:,i].tolist())
                    exist_id.update(id_from_real[real_i])
                    id_to_real[list(id_from_real[real_i])] = real_i
                    real_i = real_i + 1
            self.K_real = real_i
            id_to_real = id_to_real.unsqueeze(0).expand_as(kidx_rot)
            # print('kidx_rot original', kidx_rot)
            kidx_rot = torch.gather(id_to_real, 1, kidx_rot)
            # print('id_from_real', id_from_real)
            # print('id_to_real', id_to_real)
            # print('self.K_real', self.K_real)
            # print('kidx_rot', kidx_rot)
            # print('self.quotient_anchors', self.quotient_anchors)
            # print('self.kernel_points', self.kernel_points)

        if self.non_sep_conv:
            if not self.gather_by_idxing:
                # self.register_buffer('kidx_ori', kidx_ori.transpose(0,1)[:,None,:,None,None].expand(-1,self.kanchor,-1,self.in_channels, self.out_channels))   # k1, a(channel), a(rot)
                # self.register_buffer('kidx_rot', kidx_rot.transpose(0,1)[:,None,:,None,None].expand(-1,self.kanchor,-1,self.in_channels, self.out_channels))   # k2, a(channel), a(rot)
                self.register_buffer('kidx_rot', kidx_rot.transpose(0,1)[:,None,:].expand(-1,self.kanchor,-1))   # k2, a(channel), a(rot)
            else:
                lin_idx = torch.arange(self.kanchor).reshape(1,-1)  # 1,a
                kidx_rot_lin = kidx_rot.transpose(0,1) * self.kanchor + lin_idx  # k, a
                kidx_rot_lin = kidx_rot_lin.reshape(-1)
                self.register_buffer('kidx_rot_lin', kidx_rot_lin)   # [k*a]
        else:
            self.register_buffer('kidx_ori', kidx_ori.transpose(0,1))   # k1, a
            # self.register_buffer('kidx_rot', kidx_rot.transpose(0,1))   # k2, a
        assert torch.max(kres_ori) < 1e-3, f"{torch.max(kres_ori)}, self.equiv_mode_kp={self.equiv_mode_kp}, self.kernel_points=\n{self.kernel_points}, \nkres_ori=\n{kres_ori}"
        assert torch.max(kres_rot) < 1e-3, f"{torch.max(kres_rot)}, \n{self.kernel_points}"
        return

    def init_permute_idxs_rots(self):
        rot_anchors = torch.einsum('aij,bjk->abik', self.anchors, self.anchors)
        if self.quotient_factor == 1:
            diff_r = torch.einsum('cij,abik->cabjk', self.anchors, rot_anchors)
            cos_value = 0.5 * (torch.einsum('cabii->cab', diff_r) - 1)
            rres_ori, ridx_ori = torch.max(cos_value, 0) # cab -> ab     # find correspinding original element for each rotated
            rres_rot, ridx_rot = torch.max(cos_value, 2) # cab -> ca     # find corresponding rotated element for each original
        else:
            all_anchors = torch.einsum('aij,bjk->abik', self.anchors, self.quotient_anchors)
            diff_r = torch.einsum('cdij,abik->cdabjk', all_anchors, rot_anchors)
            cos_value = 0.5 * (torch.einsum('cdabii->cdab', diff_r) - 1)
            cos_value, _ = torch.max(cos_value, 1)  # cdab -> cab
            rres_ori, ridx_ori = torch.max(cos_value, 0) # cab -> ab     # find correspinding original element for each rotated
            rres_rot, ridx_rot = torch.max(cos_value, 2) # cab -> ca     # find corresponding rotated element for each original

            # print('ridx_rot', ridx_rot)
            # raise ValueError

        if self.non_sep_conv:
            if self.rot_by_permute:
                if not self.gather_by_idxing:
                    # self.register_buffer('ridx_ori', ridx_ori.transpose(0,1)[None,:,:,None,None].expand(self.K,-1,-1,self.in_channels, self.out_channels))   # k, a(channel), a(rot),
                    # self.register_buffer('ridx_rot', ridx_rot[None,:,:,None,None].expand(self.K,-1,-1,self.in_channels, self.out_channels))   # k, a(channel), a(rot)
                    self.register_buffer('ridx_rot', ridx_rot[None,:,:].expand(self.K,-1,-1))   # k, a(channel), a(rot)
                else:
                    # lin_idx = torch.arange(self.kanchor).reshape(-1,1)  # r,1
                    # ridx_rot_lin = ridx_rot.transpose(0,1) + lin_idx * self.kanchor  # r, a
                    # ridx_rot_lin = ridx_rot_lin.reshape(-1)
                    # self.register_buffer('ridx_rot_lin', ridx_rot_lin)   # [r*a]

                    kridx_rot_lin = self.kidx_rot_lin.reshape(self.K, self.kanchor, 1) * self.kanchor # k, a, 1
                    ridx_rot_lin_k = ridx_rot.transpose(0,1).reshape(1, self.kanchor, self.kanchor)
                    kridx_rot_lin = (kridx_rot_lin + ridx_rot_lin_k).reshape(-1)
                    self.register_buffer('kridx_rot_lin', kridx_rot_lin)   # [r*a*a]

            else:
                if not self.gather_by_idxing:
                    self.register_buffer('ridx_ori', ridx_ori[None,:,:].expand(self.K,-1,-1))   # k, a(rot), a(channel)
                else:
                    lin_idx = torch.arange(self.kanchor).reshape(-1,1)  # a,1
                    ridx_ori_lin = ridx_ori + lin_idx * self.kanchor  # a, a
                    ridx_ori_lin = ridx_ori_lin.reshape(-1)
                    self.register_buffer('ridx_ori_lin', ridx_ori_lin)   # [a*a]
        else:
            # self.register_buffer('ridx_ori', ridx_ori.transpose(0,1))   # b, a
            # self.register_buffer('ridx_rot', ridx_rot)   # c, a
            pass

        assert torch.max((rres_ori - 1).abs()) < 1e-3, f"{torch.min(rres_ori)}, \n{self.anchors}"
        assert torch.max((rres_rot - 1).abs()) < 1e-3, f"{torch.min(rres_rot)}, \n{self.anchors}"
        return


if __name__ == '__main__':
    ### If the assertion at the end of init_permute_idxs_rots() and init_permute_idxs_kpts() are not violated, 
    ### the symmetry of the kernel points and rotation anchors is satisfied, 
    ### i.e., the rotation of the kernel points are the same as permutations of the points, 
    ### and the set of rotations are closed under multiplication (when quotient_factor = 1). 
    ### SO(2)
    ### Valid parameters: kanchor < 10, quotient_factor >= 1, k = 15
    encoder = PseudoEncoder(3, 1, 15)
    encoder = PseudoEncoder(3, 2, 15)
    ### SO(3)
    ### Valid parameter set 1: kanchor = 12, quotient_factor  = 5, k = 13
    ### Valid parameter set 2: kanchor = 60, quotient_factor  = 1, k = 13
    encoder = PseudoEncoder(12, 5, 13)
    encoder = PseudoEncoder(60, 1, 13)