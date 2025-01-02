import os
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
from smplx import SMPL
from collections import defaultdict
from matplotlib.animation import FuncAnimation, PillowWriter

FPS = 30
def get_coco_skeleton():
    # 0  - nose,
    # 1  - leye,
    # 2  - reye,
    # 3  - lear,
    # 4  - rear,
    # 5  - lshoulder,
    # 6  - rshoulder,
    # 7  - lelbow,
    # 8  - relbow,
    # 9  - lwrist,
    # 10 - rwrist,
    # 11 - lhip,
    # 12 - rhip,
    # 13 - lknee,
    # 14 - rknee,
    # 15 - lankle,
    # 16 - rankle,
    return np.array(
        [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [ 5, 11],
            [ 6, 12],
            [ 5, 6 ],
            [ 5, 7 ],
            [ 6, 8 ],
            [ 7, 9 ],
            [ 8, 10],
            [ 1, 2 ],
            [ 0, 1 ],
            [ 0, 2 ],
            [ 1, 3 ],
            [ 2, 4 ],
            [ 3, 5 ],
            [ 4, 6 ]
        ]
    )

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def axisangle2matrots(axisangle):
    '''
    :param axisangle: N*num_joints*3
    :return: N*num_joints*9
    '''
    import cv2
    batch_size = axisangle.shape[0]
    axisangle = axisangle.reshape([batch_size,-1,3])
    out_matrot = []
    for mIdx in range(axisangle.shape[0]):
        cur_axisangle = []
        for jIdx in range(axisangle.shape[1]):
            a = cv2.Rodrigues(axisangle[mIdx, jIdx:jIdx + 1, :].reshape(1, 3))[0]
            cur_axisangle.append(a)

        out_matrot.append(np.array(cur_axisangle).reshape([1,-1,9]))
    return np.vstack(out_matrot)

def compute_contact_label(feet, thr=1e-2, alpha=5):
    vel = np.zeros_like(feet[..., 0])
    label = np.zeros_like(feet[..., 0])
    
    vel[1:-1] = np.linalg.norm(feet[2:] - feet[:-2], axis=-1) / 2.0
    vel[0] = vel[1]
    vel[-1] = vel[-2]
    
    x = np.float128(alpha * (thr ** -1) * (vel - thr))
    label = 1 / (1 + np.exp(x))
    return label

def estimate_velocity(data_seq, h):
    '''
    Given some data sequence of T timesteps in the shape (T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    - h : step size
    '''
    data_tp1 = data_seq[2:]
    data_tm1 = data_seq[0:-2]
    data_vel_seq = (data_tp1 - data_tm1) / (2*h)
    return data_vel_seq

def estimate_angular_velocity(rot_seq, h):
    '''
    Given a sequence of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_velocity(rot_seq, h)
    R = rot_seq[1:-1]
    RT = np.swapaxes(R, -1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT) 

    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)

    return w

def compute_align_mats(root_orient):
    '''   compute world to canonical frame for each timestep (rotation around up axis) '''
    num_frames = root_orient.shape[0]
    # convert aa to matrices
    root_orient_mat = batch_rodrigues(torch.Tensor(root_orient).reshape(-1, 3)).numpy().reshape((num_frames, 9))

    # return compute_world2aligned_mat(torch.Tensor(root_orient_mat).reshape((num_frames, 3, 3))).numpy()

    # rotate root so aligning local body right vector (-x) with world right vector (+x)
    #       with a rotation around the up axis (+z)
    body_right = -root_orient_mat.reshape((num_frames, 3, 3))[:,:,0] # in body coordinates body x-axis is to the left
    world2aligned_mat, world2aligned_aa = compute_align_from_right(body_right)

    return world2aligned_mat

def compute_joint_align_mats(joint_seq):
    '''
    Compute world to canonical frame for each timestep (rotation around up axis)
    from the given joint seq (T x J x 3)
    '''
    left_idx = 11
    right_idx = 12

    body_right = joint_seq[:, right_idx] - joint_seq[:, left_idx]   # [T, 3]
    body_right = body_right / np.linalg.norm(body_right, axis=1)[:,np.newaxis]  # [T, 1, 3]

    world2aligned_mat, world2aligned_aa = compute_align_from_right(body_right)

    return world2aligned_mat

def compute_align_from_right(body_right):
    world2aligned_angle = np.arccos(body_right[:,0] / (np.linalg.norm(body_right[:,:2], axis=1) + 1e-8)) # project to world x axis, and compute angle
    body_right[:,2] = 0.0
    world2aligned_axis = np.cross(body_right, np.array([[1.0, 0.0, 0.0]]))

    world2aligned_aa = (world2aligned_axis / (np.linalg.norm(world2aligned_axis, axis=1)[:,np.newaxis]+ 1e-8)) * world2aligned_angle[:,np.newaxis]
    world2aligned_mat = batch_rodrigues(torch.Tensor(world2aligned_aa).reshape(-1, 3)).numpy()

    return world2aligned_mat, world2aligned_aa

def action_vis_multi_view(motion, animation_path='test.gif', joint_type='coco'):
    """
    pose3d : [T, J, 3]
    """
    skelton_info = eval(f'get_{joint_type}_skeleton')()
    
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    def init():    
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax1.view_init(azim=-45, elev=45)

        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(-1, 1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        ax2.view_init(azim=-90, elev=90)
    
    def update(frame):
        ax1.clear()
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax1.view_init(azim=-45, elev=45)
        ax1.dist = 7.5

        ax2.clear()
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(-1, 1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        ax2.view_init(azim=-90, elev=90)
        ax2.dist = 7.5

        # 현재 프레임의 18개 관절 위치 데이터
        joints = motion[frame]    # [T, J, 3]
        J = joints.shape[0]

        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
        
        ax1.scatter(x, y, z)
        for i in range(J):
            ax1.text(x[i], y[i], z[i], s=f'{str(i)}')

        for (start, end) in skelton_info :
            xs, xe = x[start], x[end]
            ys, ye = y[start], y[end]
            zs, ze = z[start], z[end]
            ax1.plot((xs, xe), (ys, ye), (zs, ze))

        ax2.scatter(x, y, z)
        for i in range(J):
            ax2.text(x[i], y[i], z[i], s=f'{str(i)}')

        for (start, end) in skelton_info :
            xs, xe = x[start], x[end]
            ys, ye = y[start], y[end]
            zs, ze = z[start], z[end]
            ax2.plot((xs, xe), (ys, ye), (zs, ze))
    
    length = motion.shape[0]
    interval = 50
    ani = FuncAnimation(fig, update, frames=length, interval=interval, repeat=False, init_func=init)
    ani.save(animation_path, writer=PillowWriter(fps=20))

# Dataset load
amass_dir = '/mnt/SKY/WHAM/dataset/parsed_data/amass.pth'
db = joblib.load(amass_dir) # 
key_param = ['pose', 'betas', 'transl', 'vid']

# Body model
smpl_model = SMPL(model_path='/mnt/SKY/V_HMR/data/base_data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', num_betas=10, ext='pkl')
smpl_model = smpl_model.eval().cuda()

JOINTS_REGRESSOR_COCO = '/mnt/SKY/V_HMR/data/base_data/J_regressor_coco.npy'
coco_regressor = np.load(JOINTS_REGRESSOR_COCO)
coco_regressor = torch.from_numpy(coco_regressor).float().cuda()

# Video를 기준으로 묶기
grouped_data = defaultdict(lambda: {key: [] for key in db if key != 'vid'})

for idx, vid_name in enumerate(db['vid']):
    for key in db:
        if key != 'vid' and key != 'db':  # 'vid'는 분류의 기준으로 사용하므로 제외
            grouped_data[vid_name][key].append(db[key][idx])

for vid_name, data in grouped_data.items():
    for key in data:
        data[key] = np.array(data[key])

for vid_name, data in grouped_data.items():
    print(f"Video: {vid_name}")
    for key, values in data.items():
        print(f"  {key}: shape {values.shape}")
    """
    Video: BioMotionLab_NTroje_rub001_0002_treadmill_slow_poses.npz
    pose: shape (419, 72)
    betas: shape (419, 10)
    transl: shape (419, 3)
    """
    root_pose = data['pose'][:, :3]     # [2, 3]
    body_pose = data['pose'][:, 3:72]   # [2, 23*3]
    shape = data['betas']
    transl = data['transl']
    T = transl.shape[0]

    N = 16
    batch_root_pose = np.array_split(root_pose, N, axis=0)
    batch_body_pose = np.array_split(body_pose, N, axis=0)
    batch_shape = np.array_split(shape, N, axis=0)
    batch_trans = np.array_split(transl, N, axis=0)

    kp3d_seq = []
    for idx, (_root_pose, _body_pose, _shape, _trans) in enumerate(zip(batch_root_pose, batch_body_pose, batch_shape, batch_trans)):
        _root_pose, _body_pose, _shape, _trans =\
            torch.Tensor(_root_pose).float(), torch.Tensor(_body_pose).float(), torch.Tensor(_shape).float(), torch.Tensor(_trans).float()
        _root_pose, _body_pose, _shape, _trans =\
            _root_pose.cuda(), _body_pose.cuda(), _shape.cuda(), _trans.cuda()
        
        body_model = smpl_model(betas=_shape.reshape(-1, 10), 
                                body_pose=_body_pose.reshape(-1, 23, 3), 
                                global_orient=_root_pose.reshape(-1, 1, 3), 
                                transl=_trans.reshape(-1, 3),
                                pose2rot=True)
        vertex = body_model.vertices                    # [2, 6890, 3]
        kp3d = torch.matmul(coco_regressor, vertex)     # [2, 17, 3]
        kp3d_seq.append(kp3d.detach().cpu().numpy())
        
        if idx % 10 == 0 : print(f'{idx}/{N}')
    
    kp3d = np.concatenate(kp3d_seq, axis=0)
    offset = kp3d[:1, [11, 12]].mean(axis=1, keepdims=True) # [1, 1, 3]
    
    kp3d -= offset
    transl -= offset[:, 0]

    # action_vis_multi_view((kp3d+transl[:, np.newaxis])[:10])
    # action_vis_multi_view(kp3d[:10])
    
    h = 1.0 / FPS
    feet = kp3d[:, [15, 16]]
    contacts = compute_contact_label(feet)  # [T, ]

    joint_vel_seq = estimate_velocity(kp3d, h)
    trans_vel_seq = estimate_velocity(transl, h)
    
    root_orient_mat = axisangle2matrots(root_pose.reshape(T, 1, 3)).reshape(T, 3, 3)
    root_orient_vel_seq = estimate_angular_velocity(root_orient_mat, h)

    pose_body_mat = axisangle2matrots(body_pose.reshape(T, 23, 3)).reshape((T, 23, 3, 3))
    pose_body_vel_seq = estimate_angular_velocity(pose_body_mat, h)

    joints_world2aligned_rot = compute_joint_align_mats(kp3d)
    joint_orient_vel_seq = -estimate_angular_velocity(joints_world2aligned_rot, h)
    joint_orient_vel_seq = joint_orient_vel_seq[:,2]    # [T]

    T = T - 2
    kp3d = kp3d[1:-1]
    feet = feet[1:-1]
    contacts = contacts[1:-1]
    root_orient_mat = root_orient_mat[1:-1]
    pose_body_mat = pose_body_mat[1:-1]
    joints_world2aligned_rot = joints_world2aligned_rot[1:-1]

    trans = transl[1:-1]
    root_pose = root_pose[1:-1]
    body_pose = body_pose[1:-1]
    shape = shape[1:-1]
    

    world2aligned_rot = compute_align_mats(root_pose)   # [T, 3, 3]
    
    output = {
        'kp3d': kp3d,
        'kp3d_vel': joint_vel_seq,
        'root_pose': root_pose,
        'body_pose': body_pose,
        'root_rotmat': root_orient_mat,
        'body_rotmat': pose_body_mat,
        'root_rotmat_vel': root_orient_vel_seq,
        'trans': trans,
        'trans_vel': trans_vel_seq,
        'shape': shape,
        'contacts': contacts,
        'world2aligned_rot': world2aligned_rot,
    }
    
    output_name = vid_name.replace('.npz', '.pt')
    output_dir = os.path.join('./trans/seq', output_name)
    joblib.dump(output, output_dir)

    # 
    # global_world2aligned_trans = np.zeros((1, 3))
    # trans2joint = np.zeros((1, 1, 3))
    # global_world2aligned_rot = world2aligned_rot[0]
    # global_world2aligned_trans[0, :2] = -trans[0, :2]
    # trans2joint[0, 0, :2] = -(kp3d[0:1, 0, :] + global_world2aligned_trans)[0, :2]
    
    # print(global_world2aligned_rot.shape, root_orient_mat.shape)
    # align_root_orient = np.matmul(global_world2aligned_rot, root_orient_mat.copy()).reshape((T, 9))
    
    # global_trans = trans.copy() + global_world2aligned_trans
    # align_trans = np.matmul(global_world2aligned_rot, global_trans.T).T

    # global_joints = kp3d.copy() + global_world2aligned_trans.reshape((1,1,3))
    # global_joints += trans2joint
    # align_joints = np.matmul(global_world2aligned_rot, global_joints.reshape((-1, 3)).T).T.reshape((T, 17, 3))
    # align_joints -= trans2joint

    # global_trans_vel = trans_vel_seq.copy()
    # align_trans_vel = np.matmul(global_world2aligned_rot, global_trans_vel.T).T

    # global_root_orient_vel = root_orient_vel_seq.copy()
    # align_root_orient_vel = np.matmul(global_world2aligned_rot, global_root_orient_vel.T).T

    # global_joints_vel = joint_vel_seq.copy()
    # align_joints_vel = np.matmul(global_world2aligned_rot, global_joints_vel.reshape((-1, 3)).T).T.reshape((T, 17, 3))

    # align_body_mat = pose_body_mat.copy()
    
    # action_vis_multi_view(align_joints[:10])
    # # action_vis_multi_view(kp3d[:10])
### 