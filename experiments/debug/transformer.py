"""
Script to test that RPEConditionalTransformer has the equivariance property, 
and that two implementations are equivalent. 
"""

import torch
import torch.nn as nn

from geotransformer.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer
import geotransformer.modules.transformer.utils_epn.anchors as L

hidden_dim = 8
num_heads = 2
dropout = None
activation_fn = 'ReLU'
na = 12

# r: rotation, a: anchor, self, cross_a_soft, self, cross_r_soft, invariant, self, cross_non_equivariant = false
blocks = ['cross_r_soft', 'self_eq', 'cross_eq', 'cross_a_soft', 'cross_a_best', 'cross_r_best']
# which one is reference, '0', 1 map to 0, dual early: cat features from two point cloud, before pooling 
align_modes = ['0', '1', 'dual_early', 'dual_late']
# permutation anchor index
alternative_impls = [True, False]

vs, v_adjs, v_level2s, v_opps, vRs = L.get_icosahedron_vertices()
trace_idx_ori, trace_idx_rot = L.get_relativeV_index()  # 60*12, 60*12 (ra)
trace_idxR_ori, trace_idxR_rot = L.get_relativeR_index(vRs) # ra, ar
trace_idxR_rot = trace_idxR_rot.swapaxes(0,1)   # ra

trace_idx_ori = torch.tensor(trace_idx_ori)
trace_idx_rot = torch.tensor(trace_idx_rot)
trace_idxR_ori = torch.tensor(trace_idxR_ori)
trace_idxR_rot = torch.tensor(trace_idxR_rot)
vRs = torch.tensor(vRs)
# print('trace_idx_ori \n', trace_idx_ori)
# print('trace_idx_rot \n', trace_idx_rot)
# print('trace_idxR_ori \n', trace_idxR_ori)
# print('trace_idxR_rot \n', trace_idxR_rot)

def check_Ridx_individual():
    R1 = vRs[1]
    R2 = vRs[2]
    R1_inv2 = torch.matmul(R1.T, R2)
    R1_inv2_2 = torch.einsum('ji,jk->ik', R1, R2)
    R_rotij = vRs[trace_idxR_rot]
    R1_inv2_3 = R_rotij[1, 2]
    idx = trace_idxR_rot[1, 2]
    R1_inv2_4 = vRs[idx]
    print("R1:", R1)
    print("R2:", R2)
    print("R1_inv2:", R1_inv2)
    print("R1_inv2_2:", R1_inv2_2)
    print("R1_inv2_3:", R1_inv2_3)
    print("R1_inv2_4:", R1_inv2_4)
    return 

def check_Ridx_all():
    Ri_inv_Rj = torch.einsum('rji,sjk->rsik', vRs, vRs)
    R_rotij = vRs[trace_idxR_rot]
    print("Ri_inv_Rj:", Ri_inv_Rj.shape)    # Ri^-1 Rj
    print("R_rotij:", R_rotij.shape)
    max_err = torch.max((Ri_inv_Rj - R_rotij).abs())
    print("max err RinvIJ:", max_err)  
    return 

def check_conjugate():
    brot_i = trace_idxR_rot[:, 1].reshape(1, -1)    # 1*r
    bori_kik = trace_idxR_ori.gather(0, brot_i) # 1*60
    vRs_conj_by_perm = vRs[bori_kik.flatten()]
    vRs_conj_by_rot = torch.einsum('rji, jk, rkl -> ril', vRs, vRs[1], vRs)
    
    print("vRs_conj_by_perm:", vRs_conj_by_perm.shape)
    print("vRs_conj_by_rot:", vRs_conj_by_rot.shape)
    max_err = torch.max((vRs_conj_by_rot - vRs_conj_by_perm).abs())
    print("max err vRs:", max_err)   
    return

def check_rot_permute(f2):
    f2_permute = f2[:, trace_idx_ori]       # branc
    f2_rot = f2[:, trace_idx_rot[1]]        # banc
    f2_rotperm = f2_rot[:, trace_idx_ori]   # branc

    brot_ik = trace_idxR_rot[1].reshape(-1,1)
    brot_kik = torch.gather(trace_idxR_rot, 1, brot_ik) # r*1
    aori_kikj = trace_idx_ori[brot_kik.reshape(-1)] # ra

    # ### use gather
    # aori_kikj = aori_kikj[None, ..., None, None]
    # aori_kikj = aori_kikj.expand_as(f2_permute)
    # f2_permrot = f2_permute.gather(2, aori_kikj)
    ### use indexing
    lin_idx_b = torch.arange(3).reshape(-1,1,1)     # bra
    lin_idx_r = torch.arange(60).reshape(1,-1,1)    # bra
    idx = lin_idx_b * 60 + lin_idx_r
    idx = idx * 12 + aori_kikj[None]    # bra
    f2_permrot = f2_permute.flatten(0,2)[idx]
    # f2_permrot2 = f2_permute[:, :, trace_idx_rot[1]]
    
    print("f2_rotperm:", f2_rotperm.shape)
    print("f2_permrot:", f2_permrot.shape)
    max_err = torch.max((f2_rotperm - f2_permrot).abs())
    print("max err v:", max_err)   
    return

break_sig = False
for block in blocks:
    if break_sig:
        break
    for align_mode in align_modes:
        if break_sig:
            break
        for alternative_impl in alternative_impls:
            if break_sig:
                break

            f1 = torch.normal(0, 1, size=(3, 12, 5, 8)) # banc
            f2 = torch.normal(0, 1, size=(3, 12, 7, 8)) # bamc
            p1 = torch.normal(0, 1, size=(3, 5, 5, 8)) # bnmc
            p2 = torch.normal(0, 1, size=(3, 7, 7, 8)) # bnmc


            # break_sig = True

            ##################################
            transformer = RPEConditionalTransformer(
                        [block], hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, 
                        na=12, align_mode=align_mode, alternative_impl=alternative_impl,
                        return_attention_scores=True,
                    )


            # o1, o2, v_0, v_1 = transformer(f2, f2, p2, p2)
            # o1, o2, atts, v_0, v_1 = transformer(f2, f2, p2, p2)
            o1, o2, atts = transformer(f1, f2, p1, p2)
            print('shape o1 o2', o1.shape, o2.shape)

            # trace_idx_ori = torch.tensor(trace_idx_ori, dtype=torch.int64)

            trace_idx_ori1 = trace_idx_ori[40]   # 12
            trace_idx_rot1 = trace_idx_rot[40]   # 12
            f2_permute = f2[:, trace_idx_rot1]

            # ### test that the indexing makes sense
            # f2_permute2 = f2_permute[:, trace_idx_ori1]
            # max_err = torch.max((f2_permute2 - f2).abs())
            # print("max err:", max_err)     # 0

            # o1_permute, o2_permute, v_permute0, v_permute1 = transformer(f2, f2_permute, p2, p2)
            o1_permute, o2_permute, atts_permute= transformer(f1, f2_permute, p1, p2)   # , v_permute0, v_permute1 

            # att_r0 = atts[0][0][:, 5]
            # att_r1inv1 = atts_permute[0][0][:, 1]
            # max_err = torch.max((att_r0 - att_r1inv1).abs())
            # print("max err v:", max_err)    # checked
            # max_err = torch.max((o1_permute - o1).abs())
            # print("max err v2:", max_err)    # no


            # v1_r0 = v_1[:, 5]
            # v1_r1inv1 = v_permute1[:, 1]
            # max_err = torch.max((v1_r0 - v1_r1inv1).abs())
            # print("max err v3:", max_err)    # checked
            
            # hidden_states = torch.einsum('bahnm,bahmc->bahnc', atts[0][0][:, 5], v_1[:, 5])
            # hidden_states_r1inv1 = torch.einsum('bahnm,bahmc->bahnc', atts_permute[0][0][:, 1], v_permute1[:, 1])
            # max_err = torch.max((hidden_states - hidden_states_r1inv1).abs())
            # print("max err v4:", max_err)    # yes


            # brot_ik = torch.tensor(trace_idxR_rot[1]).reshape(-1,1)
            # brot_kik = torch.gather(torch.tensor(trace_idxR_rot), 1, brot_ik) # r*1
            # aori_kikj = trace_idx_ori[brot_kik.reshape(-1)] # ra
            # lin_idx_b = torch.arange(3).reshape(-1,1,1)     # bra
            # lin_idx_r = torch.arange(60).reshape(1,-1,1)    # bra
            # idx = lin_idx_b * 60 + lin_idx_r
            # idx = idx * 12 + aori_kikj[None]    # bra
            # atts_lateperm = atts[0][0].flatten(0,2)[idx]
            # v_1_lateperm = v_1.flatten(0,2)[idx]
            # print("v_permute1:", v_permute1.shape)
            # print("v_1_lateperm:", v_1_lateperm.shape)
            # print("atts_permute[0][0]:", atts_permute[0][0].shape)
            # print("atts_lateperm:", atts_lateperm.shape)
            
            # # atts_lateperm = atts[0][0][:, trace_idx_rot_perm]
            # # v_1_lateperm = v_1[:, trace_idx_rot_perm]
            # max_err = torch.max((atts_permute[0][0] - atts_lateperm).abs())
            # print("max err v6:", max_err)    # no
            # max_err = torch.max((v_permute1 - v_1_lateperm).abs())
            # print("max err v7:", max_err)    # no


            # hidden_states = torch.einsum('brahnm,brahmc->bahnc', atts[0][0], v_1)
            # # hidden_states = torch.einsum('brahnm,brahmc->bahnc', atts_lateperm, v_1_lateperm)
            # hidden_states_r1inv1 = torch.einsum('brahnm,brahmc->bahnc', atts_permute[0][0], v_permute1)
            # max_err = torch.max((hidden_states - hidden_states_r1inv1).abs())
            # print("max err v5:", max_err)    # no

            try:
                o2_permute_late = o2[:, trace_idx_rot1]
            except Exception as e:
                print('block', block)
                print('align_mode', align_mode)
                print('alternative_impl', alternative_impl)
                raise ValueError(e)

            max_err = torch.max((o2_permute - o2_permute_late).abs())
            print("max err:", max_err)
            if max_err > 1e-4:
                print('block', block)
                print('align_mode', align_mode)
                print('alternative_impl', alternative_impl)
                break_sig = True