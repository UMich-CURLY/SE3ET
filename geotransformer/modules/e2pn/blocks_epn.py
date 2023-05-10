import time
import math
from turtle import forward
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
import numpy as np

import geotransformer.modules.transformer.utils_epn.anchors as L
from geotransformer.modules.kpconv.kernel_points import load_kernels
# from geotransformer.modules.e2pn.kernel_points import load_kernels
# import geotransformer.modules.e2pn.anchors as L
from geotransformer.modules.e2pn.blocks import radius_gaussian, gather, max_pool, KPConv, GroupNorm
import torch.nn.functional as F

class KPConvInterSO3(nn.Module):
    def __init__(self, kernel_size, kanchor, in_channels, out_channels, KP_extent, radius,
                KP_influence='linear', aggregation_mode='sum',
                deformable=False, modulated=False, epn_kernel=False, equiv_mode_kp=False, 
                non_sep_conv=False, rot_by_permute=False, fixed_kernel_points='center', 
                quotient_factor=1, ignore_steer_constraint=False, gather_by_idxing=False) -> None:
        super().__init__()
        
        # Save parameters
        self.kanchor = kanchor
        self.p_dim = 3  #xyz
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent  # sigma
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        self.equiv_mode_kp = equiv_mode_kp
        self.fixed_kernel_points = fixed_kernel_points #'verticals' #'center'    # 'verticals' starting from 0816
        self.rot_by_permute = rot_by_permute    # True works only if equiv_mode_kp and fixed_kernel_points are True.
        self.non_sep_conv = non_sep_conv        # True works only if rot_by_permute==True. 
        self.quotient_factor = quotient_factor  # >1 works only if non_sep_conv is True. 1 means not using quotient mode.
        self.ignore_steer_constraint = ignore_steer_constraint
        self.gather_by_idxing = gather_by_idxing # if true, saves more memory
        self.epn_kernel = epn_kernel
        if self.epn_kernel:
            # Initialize kernel points
            self.kernel_size = kernel_size
            self.kernel_points = self.init_KP()
            self.K = self.kernel_points.shape[0]
        else:
            self.K = kernel_size
            self.kernel_points = self.init_KP()
        
        # # Running variable containing deformed KP distance to input points. (used in regularization loss)
        # self.min_d2 = None
        # self.deformed_KP = None
        # self.offset_features = None
        
        if self.deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(self.kernel_size,
                                      self.p_dim,
                                      self.in_channels,
                                      self.offset_dim,
                                      KP_extent,
                                      radius,
                                      self.fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode,
                                      epn_kernel=self.epn_kernel)
            self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)
        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Initialize kernel points and anchors
        self.anchors = self.init_anchors()  # a*3*3
        if self.rot_by_permute:
            self.init_permute_idxs_kpts()
        if self.non_sep_conv:
            self.init_permute_idxs_rots()
        # if self.rot_by_permute:
        #     self.init_permute_idxs()
        
        # Initialize weights
        if self.non_sep_conv:
            if self.quotient_factor > 1 and not self.ignore_steer_constraint:
                self.weights = Parameter(torch.zeros((self.K_real, self.kanchor, in_channels, out_channels), dtype=torch.float32),
                                    requires_grad=True)
            else:
                self.weights = Parameter(torch.zeros((self.K, self.kanchor, in_channels, out_channels), dtype=torch.float32),
                                    requires_grad=True)
        else:
            self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                    requires_grad=True)

        # Reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)
        return

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
        # print('self.anchors', self.anchors.shape)
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

        assert torch.max((rres_ori - 1).abs()) < 1e-3, f"{torch.max(rres_ori)}, \n{self.anchors}"
        assert torch.max((rres_rot - 1).abs()) < 1e-3, f"{torch.max(rres_rot)}, \n{self.anchors}"
        return

    def feat_gather_by_perm(self, neighbors, neighb_inds, x):
        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            pass
        else:
            deformed_K_points = self.kernel_points

        differences = neighbors.unsqueeze(2) - deformed_K_points    # p,n,k,3
        
        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=-1)
        

        # Get Kernel point influences [n_points, n_neighbors, n_kpoints]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            # NOTE: EPN uses square dist. EPN only has linear mode.
        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # all_weights = torch.transpose(all_weights, 1, 2)  # pnka -> pkna  pnck
        
        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # all_weights = all_weights[:,:,self.kidx_ori]    # p,n,k,a

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:
            pass
        else:
            new_neighb_inds = neighb_inds

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood 
        # [n_points, a, in_fdim] -> [n_points, n_neighbors, a, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # n_points, n_kpoints, na, in_fdim (kpac)
        if self.non_sep_conv:
            weighted_features = torch.einsum('pnac,pnk->pkac', neighb_x, all_weights)
        else:
            all_weights = all_weights[:,:,self.kidx_ori]    # p,n,k,a
            weighted_features = torch.einsum('pnac,pnka->pkac', neighb_x, all_weights)
        weighted_features = weighted_features.transpose(0,1)
        return weighted_features

    def feat_gather_by_rot(self, neighbors, neighb_inds, x):
        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            pass
        else:
            deformed_K_points = self.kernel_points # nk, 3 = 15, 3
        
        # rotation (60(na)*3*3, 24(nk)*3 ->60na*3*24nk->nk*na*3)
        rotated_kernels = torch.matmul(self.anchors, deformed_K_points.transpose(0,1)).permute(2,0,1).contiguous() # nk, na, 3 = 15 x 60 x 3

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, na, dim]
        neighbors = neighbors.unsqueeze(2).unsqueeze(2)
        differences = neighbors - rotated_kernels
        
        # Get the square distances [n_points, n_neighbors, n_kpoints, na]
        sq_distances = torch.sum(differences ** 2, dim=-1)
        

        # Get Kernel point influences [n_points, n_neighbors, n_kpoints, na]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            # NOTE: EPN uses square dist. EPN only has linear mode.
        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # all_weights = torch.transpose(all_weights, 1, 2)  # pnka -> pkna  pnck
        
        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:
            pass
        else:
            new_neighb_inds = neighb_inds # N x nn

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0) # N+1 x a x dim

        # Get the features of each neighborhood 
        # [n_points, a, in_fdim] -> [n_points, n_neighbors, a, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        if self.non_sep_conv:
            weighted_features = torch.einsum('pnac,pnkb->pkbac', neighb_x, all_weights) # kb aere the rotated kernel points
        else:
            # weighted_features: n_points, n_kpoints, na, in_fdim (pkac) = 20000, 15, 60, 1
            # neighb_x: n_points, n_neighbors, na, in_fdim (pnac) = 20000, 26, 60, 1
            # all_weights: n_points, n_neighbors, n_kpoints, na (pnka) = 20000, 26, 15, 60
            weighted_features = torch.einsum('pnac,pnka->pkac', neighb_x, all_weights)
        weighted_features = weighted_features.transpose(0,1)
        return weighted_features

    def forward(self, q_pts, s_pts, neighb_inds, x):
        """q_pts: """
        
        ###################
        # Offset generation
        ###################
        if self.deformable:
            pass
        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        # s_pts: n*dim, neighb_inds: n*nn, neighbors: n*nn*dim (use multi-dim index on a single dim)
        neighbors = s_pts[neighb_inds, :] # N x nn x dim, nn=26

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        if self.rot_by_permute:
            weighted_features = self.feat_gather_by_perm(neighbors, neighb_inds, x)
        else:
            weighted_features = self.feat_gather_by_rot(neighbors, neighb_inds, x)
        # if not torch.allclose(weighted_features, weighted_features2, 1e-2, 1e-2):
        #     print("rot:", weighted_features[:3,:2])
        #     print("prm:", weighted_features2[:3,:2])
        #     print("rel dif max:", torch.max((weighted_features-weighted_features2).abs()))
        #     print("self.kernel_points", self.kernel_points)
        #     raise ValueError('Feature by permutation and rotation not the same!')
        # else:
        #     raise ValueError('Feature by permutation and rotation are the same!')
        
        # Apply modulations
        if self.deformable and self.modulated:
            pass

        if self.non_sep_conv:
            if self.rot_by_permute:
                # weighted_features # k,p,a,c1
                # self.weights # k, a, in, out
                if not self.gather_by_idxing:
                    ### impl 1: using gather
                    weights = self.weights.unsqueeze(2).expand(-1,-1,self.kanchor,-1,-1) # k, a, r, in, out
                    weights = weights.gather(0, self.kidx_rot[...,None,None].expand(-1,-1,-1,self.in_channels, self.out_channels))
                    weights = weights.gather(1, self.ridx_rot[...,None,None].expand(-1,-1,-1,self.in_channels, self.out_channels))
                    output_features = torch.einsum('kpac,karcd->prd', weighted_features, weights)
                else:
                    ### impl 2: indexing on a single dimension, twice, which saves memory
                    weights = self.weights.unsqueeze(1).expand(-1,self.kanchor,-1,-1,-1) # k, r, a, in, out
                    # weights = weights.flatten(0,1)[self.kidx_rot_lin].reshape_as(weights)
                    # weights = weights.flatten(1,2)[:,self.ridx_rot_lin].reshape_as(weights)
                    ### impl 3: indexing on a single dimension, once, which saves more memory
                    weights = weights.flatten(0,2)[self.kridx_rot_lin].reshape_as(weights)
                    output_features = torch.einsum('kpac,kracd->prd', weighted_features, weights)
                # if torch.allclose(output_features,output_features1):#, torch.max((output_features-output_features1).abs())
                #     print('non_sep_conv and rot_by_permute two impl are the same')
            else:
                # weighted_features # k,p,r,a,c1
                # self.weights # k, a, in, out
                # ridx_rot k, r, a
                if not self.gather_by_idxing:
                    ### impl 1: using gather
                    weighted_features1 = weighted_features.gather(3, self.ridx_ori[:,None, ..., None].expand_as(weighted_features))
                    output_features = torch.einsum('kprac,kacd->prd', weighted_features1, self.weights)
                else:
                    ### impl 2: indexing on a single dimension, which saves memory
                    weighted_features = weighted_features.flatten(2,3)[:,:,self.ridx_ori_lin].reshape_as(weighted_features)
                    output_features = torch.einsum('kprac,kacd->prd', weighted_features, self.weights)
                # if torch.allclose(output_features1,output_features):#, torch.max((output_features-output_features1).abs())
                #     print('non_sep_conv and not rot_by_permute two impl are the same')
        else:
            # Apply network weights [n_kpoints, n_points, out_fdim]
            weighted_features = weighted_features.flatten(1,2)  # k,pa,c1
            kernel_outputs = torch.matmul(weighted_features, self.weights)  # w: k,c1,c2
            # kpc1, kc1c2
            # kpac1, kc1c2
            output_features = torch.sum(kernel_outputs, dim=0, keepdim=False)   # pa,c2
            output_features = output_features.reshape(-1, self.kanchor, self.out_channels)  #p,a,c2

            # # normalization term. # used in predator, not in 4dPLS
            # neighbor_features_sum = torch.sum(neighb_x, dim=-1) # pnac -> pna
            # neighbor_num = torch.sum(torch.gt(neighbor_features_sum, 0.0), dim=1)  # pa
            # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))   # pa
            # output_features = output_features / neighbor_num.unsqueeze(-1)

        return output_features
        
    def __repr__(self):
        return 'KPConvInterSO3(radius: {:.2f}, extent: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(
                                                                              self.radius, self.KP_extent,
                                                                              self.in_channels,
                                                                              self.out_channels)

class KPConvIntraSO3(nn.Module):
    def __init__(self, kanchor, in_channels, out_channels) -> None:
        super().__init__()
        self.kanchor = kanchor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.anchors = self.init_anchors()
        self.intra_idx = self.init_intra_idx()
        self.K = self.intra_idx.shape[1]

        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        # NOTE: EPN use nn.init.xavier_normal_(W, gain=nn.init.calculate_gain('relu'))
        return

    def init_intra_idx(self):
        intra_idx = L.get_intra_idx(self.kanchor)
        return Parameter(torch.tensor(intra_idx, dtype=torch.long),
                         requires_grad=False)

    def init_anchors(self):
        # get so3 anchors (60x3x3 rotation matrices)
        if self.kanchor == 12 or self.kanchor == 60:
            anchors = L.get_anchors(self.kanchor)
        else:
            anchors = L.get_anchorsV()
        return Parameter(torch.tensor(anchors, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, x):
        """x: p, a, c"""
        
        np, na, nc = x.shape
        _, nk = self.intra_idx.shape

        x = x.index_select(1, self.intra_idx.flatten()).view(np,na,nk,nc).permute(2,0,1,3).contiguous().flatten(1,2) # k,pa,c1
        out_features = torch.matmul(x, self.weights).sum(0).reshape(np,na,self.out_channels)    # k,pa,c1->k,pa,c2->pa,c2
        return out_features
        

class BatchNormBlockEPN(nn.Module):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlockEPN, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm2d(in_dim, momentum=bn_momentum)
            # self.batch_norm = nn.InstanceNorm2d(in_dim, momentum=bn_momentum, affine=False)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        ### x: p*a*c
        if self.use_bn:

            x = x.unsqueeze(-1)  # p*a*c*1
            x = x.permute(3, 2, 0, 1)   # 1*c*p*a
            x = self.batch_norm(x)
            x = x.permute(2, 3, 1, 0)
            return x.squeeze(-1)
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlockEPN(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(self.in_dim,
                                                                                         self.bn_momentum,
                                                                                         str(not self.use_bn))


class UnaryBlockEPN(nn.Module):
    
    def __init__(self, in_dim, out_dim, group_norm, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super(UnaryBlockEPN, self).__init__()
        self.bn_momentum = bn_momentum
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim) #, bias=False)
        
        self.norm = GroupNormEPN(group_norm, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x


class LastUnaryBlockEPN(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        """Initialize a standard last_unary block without GN, ReLU.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        """
        super(LastUnaryBlockEPN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        x = self.mlp(x)
        return x


class GroupNormEPN(nn.Module):
    def __init__(self, num_groups, num_channels):
        r"""Initialize a group normalization block.

        Args:
            num_groups: number of groups
            num_channels: feature dimension
        """
        super(GroupNormEPN, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(self.num_groups, self.num_channels)

    def forward(self, x):
        x = x.transpose(0, 2).unsqueeze(0)  # (N, A, C) -> (B, C, A, N)
        x = self.norm(x) # (B, C, *)
        x = x.squeeze(0).transpose(0, 2)  # (B, C, A, N) -> (N, A, C)
        return x.squeeze()

class KPConvInterSO3Block(nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, sigma, group_norm, config) -> None:
        super().__init__()

        # current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv = KPConvInterSO3(config.num_kernel_points, 
                                             config.kanchor,
                                             in_dim,
                                             out_dim,
                                             sigma, 
                                             radius,
                                             config.KP_influence,
                                             config.aggregation_mode,
                                             epn_kernel=config.epn_kernel,
                                             equiv_mode_kp=config.equiv_mode_kp,
                                             non_sep_conv=config.non_sep_conv,
                                             rot_by_permute=config.rot_by_permute,
                                             fixed_kernel_points=config.fixed_kernel_points,
                                             quotient_factor=config.quotient_factor,
                                             ignore_steer_constraint=config.ignore_steer_constraint,
                                             gather_by_idxing=config.gather_by_idxing,
                                             )
        # Other opperations
        # self.batch_norm = BatchNormBlockEPN(out_dim, self.use_bn, self.bn_momentum)
        self.norm = GroupNormEPN(group_norm, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, q_pts, s_pts, neighb_inds):
        x = self.conv(q_pts, s_pts, neighb_inds, x)
        # self.batch_norm(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class KPConvIntraSO3Block(nn.Module):
    def __init__(self, block_name, in_dim, out_dim, group_norm, config) -> None:
        super().__init__()
        
        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv = KPConvIntraSO3(config.kanchor, in_dim, out_dim)
        # self.batch_norm = BatchNormBlockEPN(out_dim, self.use_bn, self.bn_momentum)
        self.norm = GroupNormEPN(group_norm, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        # self.batch_norm(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return self.leaky_relu(self.batch_norm(x))


class SimpleBlockEPN(nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, sigma, group_norm, config) -> None:
        super().__init__()
        
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.non_sep_conv = config.non_sep_conv

        self.interso3 = KPConvInterSO3Block(block_name, self.in_dim, self.out_dim, radius, sigma, group_norm, config)
        if not self.non_sep_conv:
            self.intraso3 = KPConvIntraSO3Block(block_name, self.out_dim, self.out_dim, group_norm, config)

        # Other opperations
        self.norm = GroupNormEPN(group_norm, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, q_pts, s_pts, neighb_inds):
        x = self.interso3(x, q_pts, s_pts, neighb_inds)
        if not self.non_sep_conv:
            x = self.intraso3(x)

        x = self.norm(x)
        x = self.leaky_relu(x)
        return x

class ResnetBottleneckBlockEPN(nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, sigma, group_norm, config) -> None:
        super().__init__()
        
        self.bn_momentum = config.batch_norm_momentum
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relu_end = True
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.non_sep_conv = config.non_sep_conv

        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlockEPN(in_dim, out_dim // 4, group_norm, self.bn_momentum)
        else:
            self.unary1 = nn.Identity()

        self.interso3 = KPConvInterSO3Block(block_name, out_dim // 4, out_dim // 4, radius, sigma, group_norm, config)
        if not self.non_sep_conv:
            self.intraso3 = KPConvIntraSO3Block(block_name, out_dim // 4, out_dim // 4, group_norm, config)

        # self.norm = nn.InstanceNorm2d(out_dim // 4, affine=False)
        self.norm = GroupNormEPN(group_norm, out_dim // 4)
        
        # Second upscaling mlp
        self.unary2 = UnaryBlockEPN(out_dim // 4, out_dim, group_norm, self.bn_momentum, no_relu=self.relu_end)
        
        # Shortcut optional mpl
        if in_dim != out_dim:
            self.skip_conv = UnaryBlockEPN(in_dim, out_dim, group_norm, self.bn_momentum, no_relu=self.relu_end)
        else:
            self.skip_conv = nn.Identity()
            # NOTE: no_relu is diff from Predator

    def forward(self, x, q_pts, s_pts, neighb_inds):
        skip_features = x

        x = self.unary1(x)
        x = self.interso3(x, q_pts, s_pts, neighb_inds)
        if not self.non_sep_conv:
            x = self.intraso3(x)

        x = self.norm(x)
        x = self.leaky_relu(x)
        x = self.unary2(x)

        if 'strided' in self.block_name:
            skip_features = max_pool(skip_features, neighb_inds)    # NOTE: diff from EPN

        shortcut = self.skip_conv(skip_features)
        x = x + shortcut    # NOTE: Predator does relu here but not in skip conv
        if self.relu_end:
            x = self.leaky_relu(x)
        return x

class InvOutBlockEPN(nn.Module):
    def __init__(self, block_name, in_dim, config) -> None:
        super().__init__()
        self.block_name = block_name
        self.in_dim = in_dim
        self.att_pooling = config.att_pooling #if hasattr(config, 'att_pooling') else True
        self.att_permute = config.att_permute
        self.dual_feature = config.dual_feature
        if self.att_pooling or self.att_permute:
            self.attention_layer = nn.Sequential(nn.Linear(in_dim, in_dim, bias=True), \
                                                    nn.ReLU(inplace=True), \
                                                    nn.Linear(in_dim,1,bias=True))

        if self.att_permute:
            self.kanchor = config.kanchor
            self.quotient_factor = config.quotient_factor  # >1 works only if non_sep_conv is True. 1 means not using quotient mode.
            self.anchors = self.init_anchors()  # a*3*3

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
            
            ridx_rot = ridx_rot.transpose(1, 0) # ac (ra)
            self.ridx_rot = nn.Parameter(ridx_rot, requires_grad=False)

    def init_anchors(self):
        # get so3 anchors (60x3x3 rotation matrices)
        if self.quotient_factor == 1:
            if self.kanchor == 12 or self.kanchor == 60:
                anchors = L.get_anchorsV()
            else:
                anchors = L.get_anchors(self.kanchor)
        else:
            if self.kanchor == 12 or self.kanchor == 60:
                anchors = L.get_anchorsV()
                quotient_anchors = L.get_anchorsV()
            else:
                anchors = L.get_anchors(self.kanchor * self.quotient_factor)[:self.kanchor]
                quotient_anchors = L.get_anchors(self.quotient_factor)
            self.quotient_anchors = Parameter(torch.tensor(quotient_anchors, dtype=torch.float32),
                         requires_grad=False)
        return Parameter(torch.tensor(anchors, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, x, q_pts=None, s_pts=None, neighb_inds=None):
        ### x: pac -> pc
        np, na, nc = x.shape                # p,a,c
        if self.att_pooling or self.att_permute:
            attention = self.attention_layer(x) # p,a,1
            attention_w = F.softmax(attention, 1)   # p,a,1
            if self.att_pooling:
                x_fusion = (x * attention_w).sum(1) # p,c
            else:
                x_permute = x[:, self.ridx_rot] #pac -> prac
                x_permute = x_permute.flatten(2) # pr[ac]
                x_fusion = (x_permute * attention_w).sum(1)
            if self.dual_feature:
                x_fusion_max = torch.max(x, dim=1, keepdim=False)[0]
                return x_fusion_max, x_fusion
        else:
            x_fusion = torch.max(x, dim=1, keepdim=False)[0]
            # x_fusion = torch.mean(x, dim=1, keepdim=False)
        return x_fusion

class LiftBlockEPN(nn.Module):
    def __init__(self, block_name, in_dim, config):
        
        super().__init__()
        self.block_name = block_name
        self.in_dim = in_dim
        self.kanchor = config.kanchor

    def forward(self, x):
        np, nc = x.shape                
        x = x.unsqueeze(1).expand(-1, self.kanchor, -1) # (N, C) -> (N, A, C)
        return x