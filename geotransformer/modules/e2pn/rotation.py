import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as sciR
import math

class RigidMatrix():
    def __init__(self, data):
        assert data.shape == (4,4)
        self._data = data

    @classmethod
    def fromRt(cls, R, t):
        M = np.zeros([4,4])
        M[:3, :3] = R
        M[-1, :3] = t
        return RigidMatrix(M)

    @property
    def R(self):
        return self._data[:3,:3]

    @property
    def t(self):
        return self._data[-1,:3]

    @property
    def data(self):
        return self._data

    @property
    def T(self):
        return self.inverse()

    def __add__(self, other):
        return RigidMatrix(self.data + other.data)

    def __mul__(self, other):
        return RigidMatrix(self.data @ other.data)

    def inverse(self):
        return RigidMatrix.fromRt(self.R.T, -self.t)

def rotationMatrixToEulerAngles(R) :

    # assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def rand_rotation_matrix(deflection=1.0, randnums=None, makeT=False):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)

    if makeT:
        T = np.identity(4)
        T[0, :3] = M[0]
        T[1, :3] = M[1]
        T[2, :3] = M[2]
        return T
    else:
        return M

def get_adjmatrix_trimesh_vtx(mesh, gsize=None):
    vertices = mesh.vertices    # 12,3
    vtx_adj = mesh.edges    # 60*2 (each edge has the reverse pair as well)
    v_neighbors = mesh.vertex_neighbors # 12,5 ndarray

    level_2s = []
    opposites = []
    for i in range(vertices.shape[0]):
        counted = []
        counted.append(i)
        vns = v_neighbors[i]
        counted.extend(vns)
        level_2 = []
        for vn in vns:
            vnns = v_neighbors[vn]
            for vnn in vnns:
                if vnn not in counted:
                    level_2.append(vnn)
                    counted.append(vnn)
        level_2s.append(level_2)
        for j in range(vertices.shape[0]):
            if j not in counted:
                opposites.append(j)
                break
    
    level_2s = np.array(level_2s)   # 12,5
    opposites = np.array(opposites) # 12
            
    return v_neighbors, level_2s, opposites

# functions for so3 sampling
def get_adjmatrix_trimesh(mesh, gsize=None):
    # (20, 3). 20 faces. Each has 3 vertices. 
    face_idx = mesh.faces
    # (30, 2). 30 edges. Each has 2 adjacent faces. It contains all adjacent face pairs. 
    face_adj = mesh.face_adjacency
    adj_idx = []
    binary_swap = np.vectorize(lambda a: 1 if a == 0 else 0)
    for i, fidx in enumerate(face_idx):
        # for each face, find all pairs that this face appears. Then the other face in the pair is an adjacent face.
        fid = np.argwhere(face_adj == i)
        fid[:,1] = binary_swap(fid[:,1])
        adj_idx.append(face_adj[tuple(np.split(fid, 2, axis=1))].T)

    # (20, 3). 20 faces. Each has 3 adjacent faces. 
    face_adj =  np.vstack(adj_idx).astype(np.int32)

    if gsize is None:
        return face_adj
    else:
        # Padding with in-plane rotation neighbors
        na = face_adj.shape[0]  # 20
        # neighbor rotations from neighbor faces (3*3=9)
        R_adj = (face_adj * gsize)[:,None].repeat(gsize, axis=1).reshape(-1,3)  # 20*3 -> 20*1*3 ->20*3*3 -> 60*3
        R_adj = np.tile(R_adj,[1,gsize]) + np.arange(gsize).repeat(3)[None].repeat(na*gsize, axis=0)    # 60*9
        # neighbor rotations from the same face (3)
        rp = (np.arange(na) * gsize).repeat(gsize)[..., None].repeat(gsize,axis=1)  # 20->60->60*1->60*3
        rp = rp + np.arange(gsize)[None].repeat(na*gsize,axis=0)    # 60*3
        R_adj = np.concatenate([R_adj, rp], axis=1)     # 60*12
        return R_adj

def get_so3_from_anchors_np(face_normals, gsize=3):
    # alpha, beta
    na = face_normals.shape[0]
    sbeta = face_normals[...,-1]
    cbeta = (1 - sbeta**2)**0.5
    calpha = face_normals[...,0] / cbeta
    salpha = face_normals[...,1] / cbeta

    if gsize==5:
        calpha = np.where(np.isnan(calpha), np.ones_like(calpha), calpha)
        salpha = np.where(np.isnan(salpha), np.zeros_like(salpha), salpha)

    # gamma
    gamma = np.linspace(0, 2 * np.pi, gsize, endpoint=False, dtype=np.float32)
    gamma = -gamma[None].repeat(na, axis=0)

    # Compute na rotation matrices Rx, Ry, Rz
    Rz = np.zeros([na, 9], dtype=np.float32)
    Ry = np.zeros([na, 9], dtype=np.float32)
    Rx = np.zeros([na, gsize, 9], dtype=np.float32)
    Rx2 = np.zeros([na, gsize, 9], dtype=np.float32)

    # see xyz convention in http://mathworld.wolfram.com/EulerAngles.html
    # D matrix
    Rz[:,0] = calpha
    Rz[:,1] = salpha
    Rz[:,2] = 0
    Rz[:,3] = -salpha
    Rz[:,4] = calpha
    Rz[:,5] = 0
    Rz[:,6] = 0
    Rz[:,7] = 0
    Rz[:,8] = 1

    # C matrix
    Ry[:,0] = cbeta
    Ry[:,1] = 0
    Ry[:,2] = sbeta
    Ry[:,3] = 0
    Ry[:,4] = 1
    Ry[:,5] = 0
    Ry[:,6] = -sbeta
    Ry[:,7] = 0
    Ry[:,8] = cbeta

    # B Matrix
    Rx[:,:,0] = 1
    Rx[:,:,1] = 0
    Rx[:,:,2] = 0
    Rx[:,:,3] = 0
    Rx[:,:,4] = np.cos(gamma)
    Rx[:,:,5] = np.sin(gamma)
    Rx[:,:,6] = 0
    Rx[:,:,7] = -np.sin(gamma)
    Rx[:,:,8] = np.cos(gamma)

    # padding = 60  # hardcoded for gsize=3
    padding = 2 * np.pi / gsize / 2 # adaptive to gsize
    Rx2[:,:,0] = 1
    Rx2[:,:,1] = 0
    Rx2[:,:,2] = 0
    Rx2[:,:,3] = 0
    Rx2[:,:,4] = np.cos(gamma+padding) #/180*np.pi
    Rx2[:,:,5] = np.sin(gamma+padding) #/180*np.pi
    Rx2[:,:,6] = 0
    Rx2[:,:,7] = -np.sin(gamma+padding) #/180*np.pi
    Rx2[:,:,8] = np.cos(gamma+padding) #/180*np.pi

    Rz = Rz[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Ry = Ry[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Rx = Rx.reshape(na*gsize,3,3)
    Rx2 = Rx2.reshape(na*gsize,3,3)

    # R = BCD
    Rxy = np.einsum('bij,bjh->bih', Rx, Ry)
    Rxy2 = np.einsum('bij,bjh->bih', Rx2, Ry)
    Rs1 = np.einsum('bij,bjh->bih', Rxy, Rz)
    Rs2 = np.einsum('bij,bjh->bih', Rxy2, Rz)

    z_val = (face_normals[:, -1])[:, None].repeat(gsize, axis=1).reshape(na*gsize, 1, 1)
    # import ipdb; ipdb.set_trace()
    if gsize == 3:
        Rs = Rs1*(np.abs(z_val+0.79)<0.01)+Rs2*(np.abs(z_val+0.19)<0.01)+\
            Rs1*(np.abs(z_val-0.19)<0.01)+Rs2*(np.abs(z_val-0.79)<0.01)
        # -0.7947, -0.1876, 0.1876, 0.7967
        # each will make only one of the four conditions true
    elif gsize == 5:
        Rs = Rs1*(np.abs(z_val+1)<0.01)+Rs2*(np.abs(z_val+0.447)<0.01)+\
            Rs1*(np.abs(z_val-0.447)<0.01)+Rs2*(np.abs(z_val-1)<0.01)
    else:
        raise NotImplementedError('gsizee other than 3 (for faces) or 5 (for vertices) are not supported: %d'%gsize)
    return Rs

# def get_so3_from_anchors_np(face_normals, gsize=3):
#     # alpha, beta
#     na = face_normals.shape[0]
#     gamma = np.linspace(0, 2 * np.pi, gsize, endpoint=False, dtype=np.float32)

#     quat = np.zeros((na * gsize, 4))
#     for i in range(na):
#         for j in range(gsize):
#             quat[i*3+j, :3] = face_normals[i] * np.sin(gamma[j] / 2)
#             quat[i*3+j, 3] = np.cos(gamma[j] / 2)

#     Rs = sciR.from_quat(quat).as_matrix()

#     return Rs

def icosahedron_trimesh_to_vertices(mesh_path):
    mesh = trimesh.load(mesh_path)  # trimesh 3.9 does not work. need 3.2
    mesh.fix_normals()
    vs = mesh.vertices  # each vertex is of norm 1

    # the 5 rotation matrices for each of the 12 vertices
    Rs = get_so3_from_anchors_np(vs, gsize=5)
    # Rs = Rs.reshape(vs.shape[0], 5, 3, 3)
    # the index of the opposite vertex and the two five-vertex-ring for each vertex
    v_adjs, v_level2s, v_opps = get_adjmatrix_trimesh_vtx(mesh)
    return vs, v_adjs, v_level2s, v_opps, Rs

def get_relativeV_index(Rs, vs):
    # the permutation of the 12 vertices under the 60 rotations
    # Rs: 60*3*3, vs: 12*3
    print("Rs.shape", Rs.shape, vs.shape)
    incr_r = np.einsum('dij,aj->dai', Rs, vs) # drotation*anchor, 60*12*3
    incr_r = incr_r[:,:,None]   # 60*12*1*3
    ori_vs = vs[None,None]      # 1*1*12*3

    diff_r = incr_r - ori_vs  # 60*12(rot)*12(ori)*3
    trace = (diff_r**2).sum(-1)  # 60*12*12
    trace_idx_ori = np.argmin(trace,axis=2) # find correspinding original element for each rotated
    trace_idx_rot = np.argmin(trace,axis=1) # find corresponding rotated element for each original

    return trace_idx_ori, trace_idx_rot

def icosahedron_so3_trimesh(mesh_path, gsize=3, use_quats=False):
    # 20 faces, 12 vertices
    # root = vgtk.__path__[0]
    # mesh_path = os.path.join(root, 'data', 'anchors/sphere12.ply')
    mesh = trimesh.load(mesh_path)  # trimesh 3.9 does not work. need 3.2
    mesh.fix_normals()
    # (20, 3) 20 faces, each with 3 vertices. 
    face_idx = mesh.faces
    # (20, 3) 20 faces, each with 3-vector normal.
    face_normals = mesh.face_normals

    fix_angle = np.arctan(face_normals[9, 2] / face_normals[9, 0])
    fix_rot = np.array([[np.cos(fix_angle),  0,  np.sin(fix_angle)],
                        [0,                  1,  0],
                        [-np.sin(fix_angle), 0, np.cos(fix_angle)]])
    # face_normals = face_normals @ fix_rot.T

    na = face_normals.shape[0]
    # gsize = 3

    # 60x3x3
    Rs = get_so3_from_anchors_np(face_normals, gsize=gsize) # .reshape(na, gsize, 3, 3)
    # align the rotations to a reference such that Rs[29] is identity. 
    Rs = np.einsum('bij,kj', Rs, Rs[29])
    # 60x12 indices of adjacent rotations
    R_adj = get_adjmatrix_trimesh(mesh, gsize)

    # 60x12x3x3
    grouped_R = np.take(Rs, R_adj, axis=0)

    # relative_Rs = np.einsum('bkij,bjh->bkih', grouped_R, np.transpose(Rs,(0,2,1)))

    # # 12x3x3
    # canonical_R = relative_Rs[0]
    # nn = canonical_R.shape[0]

    # # 60x12x3x3
    # ordered_R = np.einsum('kij,bkjh->bkih',canonical_R, Rs[:,None].repeat(nn, axis=1))

    ################
    # the relative rotations for all 60 rotations are the same. Therefore only need to calculate on one. 
    relative_Rs = np.einsum('kjh,lh->kjl', grouped_R[0], Rs[0]) # 12x3x3
    # relative_Is = np.einsum('', relative_Rs, )
    # the order of neighbors are not guaranteed to be consistent across the 60 rotations, 
    # therefore use the order of neighbors in the first rotation to regenerate all neighboring rotations. 
    ordered_R = np.einsum('kmj,bji->bkim', relative_Rs, Rs) # 60x12x3x3
    # ordered_R = np.einsum('kmj,bji->bkmi', relative_Rs, Rs) # 60x12x3x3
    # ordered_R = np.einsum('kmj,bij,kli->bkml', relative_Rs, Rs, relative_Rs) # 60x12x3x3
    # ordered_R = np.einsum('bml,kmj,bij->bkli', Rs, relative_Rs, Rs)
    # ordered_R = np.einsum('bkmi,kjm,bkjl->bkli', ordered_R, relative_Rs, ordered_R)
    canonical_R = None

    #################

    # grouped_R = np.einsum('kij,bkjh->bkih', relative_Rs, grouped_R)

    # 60x12x1x3x3, 60x1x12x3x3 -> 60x12x12x3x3 -> 60x12x1 argmin diff
    # 60x12x1x3x3
    tiled_ordr = np.expand_dims(ordered_R,axis=2)

    ### # 60x12x60x3x3
    diff_r = np.einsum('bkgij,chj->bkcih', tiled_ordr, Rs)

    ## stop using grouped_R
    # tiled_grpr = np.expand_dims(grouped_R,axis=1)
    # # 60x12x12x3x3
    # diff_r = np.einsum('bkgij,bkghj->bkgih', tiled_ordr, tiled_grpr)
    ## stop end

    # cos() = 0.5(tr(R) - 1), 60x12x60
    trace = 0.5 * (np.einsum('bkgii->bkg', diff_r) - 1)
    # 60x12 true index wrt ordered_R
    # max cos -> smallest diff -> the index of self, 60x12
    trace_idx = np.argmax(trace,axis=2)

    # import ipdb; ipdb.set_trace()
    reverse_Rs_tmp = np.einsum('nij,mjk->nmji', Rs, Rs)     # 60*60*3*3
    reverse_Rs_tmp_sum = reverse_Rs_tmp.sum(2).sum(2)       # 60*60
    ### equivalent to this (why?)
    # reverse_Rs_tmp = np.einsum('nij,mjk->nmik', Rs, Rs)     # 60*60*3*3
    # reverse_Rs_tmp_sum = 0.5 * (np.einsum('bkii->bk', reverse_Rs_tmp) - 1)

    reverse_Rs_idx = np.argmax(reverse_Rs_tmp_sum, axis=1)  # 60
    trace_idx = trace_idx[reverse_Rs_idx]                   # 60*12

    # r_tmp = np.matmul(Rs[0], Rs[reverse_Rs_idx[0]]) 
    # print(r_tmp)
    # identity (so the reverse_Rs_idx seems to find the index of the inverse rotation)
    # In convolution, we take the input rotation neighborhood around the inverse of the kernel rotation, 
    # therefore here trace_idx are taken after reverse_Rs_idx. 
    # It is used to gather feature rotation channels. (see intra_so3conv_grouping)

    use_idx = [2,3,6,9]
    new_trace_idx = np.zeros([trace_idx.shape[0], len(use_idx)], dtype=np.int32)

    for i in range(trace_idx.shape[0]):
        # trace_idx[i] = R_adj[i][trace_idx[i]]
        new_trace_idx[i] = trace_idx[i,use_idx]

    # ---------------- DEBUG ONLY -------------------------
    # np.set_printoptions(precision=2, suppress=True)
    # print(sciR.from_matrix(np.matmul(ordered_R[1], Rs[1].T)).as_quat())
    # for i in range(30):
    #     print(sciR.from_matrix(np.matmul(Rs[trace_idx[i]], Rs[i].T)).as_quat())
    # import ipdb; ipdb.set_trace()
    # -----------------------------------------------------
    # trace_idx = np.arange(60)[:,None].astype(np.int32)

    if use_quats:
        Rs = sciR.from_matrix(Rs).as_quat()
    # Rs = np.transpose(Rs, [0,2,1])

    reverse_trace_idx = np.zeros_like(new_trace_idx)
    for i in range(new_trace_idx.shape[1]):
        for j in range(new_trace_idx.shape[0]):
            reverse_trace_idx[new_trace_idx[j,i], i] = j

    #################### DEBUG ###########################
    # for i in range(100):
    #     randR = sciR.random().as_matrix()
    #     traces, nns = rotation_distance_np(randR, Rs)
    #     nnidx = np.argsort(-traces)[:5]
    #     print(traces[nnidx])
    # import ipdb; ipdb.set_trace()
    ####################################################

    return Rs, trace_idx, canonical_R
    # return Rs, trace_idx, canonical_R

def get_relativeR_index(Rs):
    incr_r = np.einsum('dij,ajk->daik', Rs, Rs) # drotation*anchor
    diff_r = np.einsum('bij,daik->bdajk', Rs, incr_r)
    trace = 0.5 * (np.einsum('bdaii->bda', diff_r) - 1)
    trace_idx_ori = np.argmax(trace,axis=0) # bda -> da     # find correspinding original element for each rotated
    trace_idx_rot = np.argmax(trace,axis=2) # bda -> bd     # find corresponding rotated element for each original

    return trace_idx_ori, trace_idx_rot

def test_relativeR_index(trace_idx_ori, trace_idx_rot, Rs):
    for i in range(60):
        for j in range(60):
            # Rs[i] @ Rs[j] == Rs[trace_idx_ori[i,j]]
            diff_r_cos = 0.5 * (np.trace(Rs[i] @ Rs[j] @ Rs[trace_idx_ori[i,j]].swapaxes(0,1)) - 1)
            assert math.isclose(diff_r_cos, 1, rel_tol=1e-5, abs_tol=1e-5), f"trace_idx_ori {i} {j} {diff_r_cos}"
            # Rs[i] == Rs[j] @ Rs[trace_idx_rot[i,j]]
            diff_r_cos = 0.5 * (np.trace(Rs[j] @ Rs[trace_idx_rot[i,j]] @ Rs[i].swapaxes(0,1)) - 1)
            assert math.isclose(diff_r_cos, 1, rel_tol=1e-5, abs_tol=1e-5), f"trace_idx_rot {i} {j} {diff_r_cos}"
    return

'''
    (B)x3x3, Nx3x3 -> dist, idx
'''
def rotation_distance_np(r0, r1):
    '''
    tip: r1 is usally the anchors
    '''
    if r0.ndim == 3:
        bidx = np.zeros(r0.shape[0]).astype(np.int32)
        traces = np.zeros([r0.shape[0], r1.shape[0]]).astype(np.int32)
        for bi in range(r0.shape[0]):
            diff_r = np.matmul(r1, r0[bi].T)
            traces[bi] = np.einsum('bii->b', diff_r)
            bidx[bi] = np.argmax(traces[bi])
        return traces, bidx
    else:
        # diff_r = np.matmul(r0, r1.T)
        # return np.einsum('ii', diff_r)

        diff_r = np.matmul(np.transpose(r1,(0,2,1)), r0)
        traces = np.einsum('bii->b', diff_r)

        return traces, np.argmax(traces), diff_r

'''
Acknowledgement: Zhou Yi
https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py
'''
import torch

#quaternion batch*4
def compute_rotation_matrix_from_quaternion(quaternion):
    # batch*n
    def normalize_vector(v, return_mag=False):
        batch=v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
        v = v/v_mag
        if(return_mag==True):
            return v, v_mag[:,0]
        else:
            return v

    batch= quaternion.shape[0]
    quat = normalize_vector(quaternion).contiguous()

    qw = quat[...,0].contiguous().view(batch, 1)
    qx = quat[...,1].contiguous().view(batch, 1)
    qy = quat[...,2].contiguous().view(batch, 1)
    qz = quat[...,3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw

    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3

    return matrix

#euler_sin_cos batch*6
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler_sin_cos(euler_sin_cos):
    batch=euler_sin_cos.shape[0]

    s1 = euler_sin_cos[:,0].view(batch,1)
    c1 = euler_sin_cos[:,1].view(batch,1)
    s2 = euler_sin_cos[:,2].view(batch,1)
    c2 = euler_sin_cos[:,3].view(batch,1)
    s3 = euler_sin_cos[:,4].view(batch,1)
    c3 = euler_sin_cos[:,5].view(batch,1)

    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3

    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3

    return matrix

def compute_rotation_matrix_from_ortho6d(ortho6d):
    def normalize_vector(v, return_mag=False):
        batch=v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
        v = v/v_mag
        if(return_mag==True):
            return v, v_mag[:,0]
        else:
            return v
    def cross_product( u, v):
        batch = u.shape[0]
        #print (u.shape)
        #print (v.shape)
        i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
        j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
        k = u[:,0]*v[:,1] - u[:,1]*v[:,0]

        out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3

        return out

    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3

    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

# torch function
def so3_mean(Rs, weights=None):
        """Get the mean of the rotations.
        Parameters
        ----------
        Rs: (B,N,3,3)
        weights : array_like shape (B,N,), optional
            Weights describing the relative importance of the rotations. If
            None (default), then all values in `weights` are assumed to be
            equal.
        Returns
        -------
        mean R: (B,3,3)
        -----
        The mean used is the chordal L2 mean (also called the projected or
        induced arithmetic mean). If ``p`` is a set of rotations with mean
        ``m``, then ``m`` is the rotation which minimizes
        ``(weights[:, None, None] * (p.as_matrix() - m.as_matrix())**2).sum()``.

        """
        nb, na, _, _ = Rs.shape
        mask = torch.Tensor([[0,0,0],[0,0,0],[0,0,1]]).float().to(Rs.device)
        mask2 = torch.Tensor([[1,0,0],[0,1,0],[0,0,0]]).float().to(Rs.device)
        mask = mask[None].expand(nb, -1, -1).contiguous()
        mask2 = mask2[None].expand(nb, -1, -1).contiguous()

        if weights is None:
            weights = 1.0
        else:
            weights = weights[:,:,None,None]

        Ce = torch.sum(weights * Rs, dim=1)
        cu, cd, cv = torch.svd(Ce)
        cvT = cv.transpose(1,2).contiguous()
        dets = torch.det(torch.matmul(cu, cvT))

        D = mask * dets[:,None,None] + mask2
        return torch.einsum('bij,bjk,bkl->bil', cu, D, cvT)

def label_relative_rotation_np(anchors, T):
    T_from_anchors = np.einsum('abc,bj,ijk -> aick', anchors, T, anchors)
    label = np.argmax(np.einsum('abii->ab', T_from_anchors),axis=1)
    idxs = np.vstack([np.arange(label.shape[0]), label]).T
    R_target = T_from_anchors[idxs[:,0], idxs[:,1]]
    return R_target, label
