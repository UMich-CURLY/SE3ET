

# initialize so3 sampling
import numpy as np
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

# ### when debugging
# import sys
# sys.path.append(os.path.join(cur_dir,'../../') )   # if vgtk is only build_ext but not installed

import geotransformer.modules.transformer.utils_epn.rotation as fr

GAMMA_SIZE = 3
# ROOT = vgtk.__path__[0]
ANCHOR_PATH = os.path.join(cur_dir, 'sphere12.ply')

vs, v_adjs, v_level2s, v_opps, vRs = fr.icosahedron_trimesh_to_vertices(ANCHOR_PATH)    # 12*3, each vertex is of norm 1

Rs, R_idx, canonical_relative = fr.icosahedron_so3_trimesh(ANCHOR_PATH, GAMMA_SIZE)

vs_tetra, v_adjs_tetra, vRs_tetra, ecs_tetra, face_normals_tetra = fr.tetrahedron_trimesh_to_vertices()

vs_octa, v_adjs_octa, v_opps_octa, vRs_octa, ecs_octa, face_normals_octa = fr.octahedron_trimesh_to_vertices()

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

def get_anchorsV(tetra=False):
    """return 60*3*3 matrix as rotation anchors determined by the symmetry of icosahedron vertices"""
    if tetra:
        return vRs_tetra.copy()
    else:
        return vRs.copy()

def get_anchorsV12(tetra=False):
    """return 12*3*3 matrix as the section (representative rotation) of icosahedron vertices. 
    For each vertex on the sphere (icosahedron) (coset space S2 = SO(3)/SO(2)), 
    pick one rotation as its representation in SO(3), which is also called a section function (G/H -> G)"""
    if tetra:
        return vRs_tetra.reshape(4, 3, 3, 3)[:,0].copy()    # 4*3*3
    else:
        return vRs.reshape(12, 5, 3, 3)[:,0].copy()    # 12*3*3

def get_anchorsV24():
    """return 24*3*3 matrix as the section (representative rotation) of octahedron vertices. 
    For each vertex on the sphere (octahedron) (coset space S2 = SO(3)/SO(2)), 
    pick one rotation as its representation in SO(3), which is also called a section function (G/H -> G)"""
    
    return vRs_octa.reshape(6, 4, 3, 3)[:,0].copy()    # 6*3*3

def get_icosahedron_vertices():
    return vs.copy(), v_adjs.copy(), v_level2s.copy(), v_opps.copy(), vRs.copy()

def get_tetrahedron_vertices():
    return vs_tetra.copy(), v_adjs_tetra.copy(), vRs_tetra.copy(), ecs_tetra.copy(), face_normals_tetra.copy()

def get_octahedron_vertices():
    return vs_octa.copy(), v_adjs_octa.copy(), vRs_octa.copy(), ecs_octa.copy(), face_normals_octa.copy()

def get_relativeV_index():
    trace_idx_ori, trace_idx_rot = fr.get_relativeV_index(vRs, vs)
    return trace_idx_ori, trace_idx_rot

def get_relativeR_index(R):
    return fr.get_relativeR_index(R)

def icosahedron_align_vs_to_Rs(vRs, vs):
    v1 = vs[0]
    vs_r = np.einsum('bij,j', vRs, v1)
    nr = vRs.shape[0]
    vs_uniq = []
    for i in range(nr):
        exist = False
        for v_uniq in vs_uniq:
            if np.allclose(v_uniq, vs_r[i], rtol=1e-02, atol=1e-03):
                exist = True
                break
        if not exist:
            vs_uniq.append(vs_r[i])
    vs_uniq = np.array(vs_uniq)
    return vs_uniq, vs_r