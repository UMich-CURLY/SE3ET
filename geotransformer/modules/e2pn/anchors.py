# initialize so3 sampling
import numpy as np
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

### when debugging
# import sys
# sys.path.append(os.path.join(cur_dir,'../../../') )   # if vgtk is only build_ext but not installed

import geotransformer.modules.e2pn.rotation as fr

GAMMA_SIZE = 3
# ROOT = vgtk.__path__[0]
ANCHOR_PATH = os.path.join(cur_dir, 'anchors_ply/sphere12.ply')

vs, v_adjs, v_level2s, v_opps, vRs = fr.icosahedron_trimesh_to_vertices(ANCHOR_PATH)    # 12*3, each vertex is of norm 1

Rs, R_idx, canonical_relative = fr.icosahedron_so3_trimesh(ANCHOR_PATH, GAMMA_SIZE)

def select_anchor(anchors, k):
    if k == 1:
        return anchors[29][None]
    elif k == 20:
        return anchors[::3]
    elif k == 40:
        return anchors.reshape(20,3,3,3)[:,:2].reshape(-1,3,3)
    else:
        return anchors

def anchor_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return R
def anchors_z(k):
    angles = np.linspace(0, 2*np.pi, k, endpoint=False)
    mats = np.stack([anchor_z(angle) for angle in angles], 0)   # k*3*3
    return mats

def get_anchors(k=60):
    if k > 1 and k < 10:
        return anchors_z(k)
    else:
        return select_anchor(Rs,k)

def idx_z(k):
    idx = np.arange(k)
    idxs = np.stack([(idx + i) % k for i in range(k)], 0)
    return idxs

def get_intra_idx(k=60):
    if k > 1 and k < 10:
        return idx_z(k)
    else:
        return R_idx

def get_canonical_relative():
    return canonical_relative

def get_relative_index():
    return fr.get_relativeR_index(Rs)

def get_anchorsV():
    return vRs

def get_icosahedron_vertices():
    return vs, v_adjs, v_level2s, v_opps, vRs

def get_relativeV_index():
    trace_idx_ori, trace_idx_rot = fr.get_relativeV_index(vRs, vs)
    return trace_idx_ori, trace_idx_rot

