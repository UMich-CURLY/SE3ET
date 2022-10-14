"""
Script to test that the generated rotation anchors indeed form a finite group.  
"""

import torch

import geotransformer.modules.transformer.utils_epn.anchors as L

vs, v_adjs, v_level2s, v_opps, vRs = L.get_icosahedron_vertices()
trace_idx_ori, trace_idx_rot = L.get_relativeV_index()  # 60*12, 60*12 (ra)
trace_idxR_ori, trace_idxR_rot = L.get_relativeR_index(vRs) # ra, ar
trace_idxR_rot = trace_idxR_rot.swapaxes(0,1)   # ra

trace_idx_ori = torch.tensor(trace_idx_ori)
trace_idx_rot = torch.tensor(trace_idx_rot)
trace_idxR_ori = torch.tensor(trace_idxR_ori)
trace_idxR_rot = torch.tensor(trace_idxR_rot)
vRs = torch.tensor(vRs)

print('vs\n', vs)
vs_uniq, vs_r = L.icosahedron_align_vs_to_Rs(vRs, vs)
print('vs_uniq\n', vs_uniq)
print('vs_r\n', vs_r)