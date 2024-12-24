import copy
import os
import json
import logging
import datetime
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

REPR_LIST = ['root_rot_angle', 'root_rot_angle_vel', 'root_l_pos', 'root_l_vel', 'root_height', # joint-based traj
             'smplx_rot_6d', 'smplx_rot_vel', 'smplx_trans', 'smplx_trans_vel',  # smplx-based traj
             'local_positions', 'local_vel',  # joint-based local pose
             'smplx_body_pose_6d',  # smplx-based local pose
             'smplx_betas',  # smplx body shape
             'foot_contact', ]

REPR_DIM_DICT = {'root_rot_angle': 1,
                 'root_rot_angle_vel': 1,
                 'root_l_pos': 2,
                 'root_l_vel': 2,
                 'root_height': 1,
                 'smplx_rot_6d': 6,
                 'smplx_rot_vel': 3,
                 'smplx_trans': 3,
                 'smplx_trans_vel': 3,
                 'local_positions': 22 * 3,
                 'local_vel': 22 * 3,
                 'smplx_body_pose_6d': 21 * 6,
                 'smplx_betas': 10,
                 'foot_contact': 4, }

def update_globalRT_for_smpl(body_param_dict, trans_to_target_origin, smpl_model=None, device=None, delta_T=None):
    '''
    input:
        body_param_dict:
        smpl_model: the model to generate smpl mesh, given body_params
        trans_to_target_origin: coordinate transformation [4,4] mat
        delta_T: pelvis location?
    Output:
        body_params with new globalR and globalT, which are corresponding to the new coord system
    '''

    ### step (1) compute the shift of pelvis from the origin
    bs = len(body_param_dict['transl'])

    if delta_T is None:
        body_param_dict_torch = {}
        for key in body_param_dict.keys():
            body_param_dict_torch[key] = torch.FloatTensor(body_param_dict[key]).to(device)
        body_param_dict_torch['transl'] = torch.zeros([bs, 3], dtype=torch.float32).to(device)
        body_param_dict_torch['global_orient'] = torch.zeros([bs, 3], dtype=torch.float32).to(device)

        smpl_out = smpl_model(**body_param_dict_torch)
        delta_T = smpl_out.joints[:,0,:] # (bs, 3,)
        delta_T = delta_T.detach().cpu().numpy()

    ### step (2): calibrate the original R and T in body_params
    body_R_angle = body_param_dict['global_orient']
    body_R_mat = R.from_rotvec(body_R_angle).as_matrix() # to a [bs, 3,3] rotation mat
    body_T = body_param_dict['transl']
    body_mat = np.zeros([bs, 4, 4])
    body_mat[:, :-1,:-1] = body_R_mat
    body_mat[:, :-1, -1] = body_T + delta_T
    body_mat[:, -1, -1] = 1

    ### step (3): perform transformation, and decalib the delta shift
    body_params_dict_new = copy.deepcopy(body_param_dict)
    trans_to_target_origin = np.expand_dims(trans_to_target_origin, axis=0)  # [1, 4]
    trans_to_target_origin = np.repeat(trans_to_target_origin, bs, axis=0)  # [bs, 4]

    body_mat_new = np.matmul(trans_to_target_origin, body_mat)  # [bs, 4, 4]
    body_R_new = R.from_matrix(body_mat_new[:, :-1,:-1]).as_rotvec()
    body_T_new = body_mat_new[:, :-1, -1]
    body_params_dict_new['global_orient'] = body_R_new.reshape(-1,3)
    body_params_dict_new['transl'] = (body_T_new - delta_T).reshape(-1,3)
    return body_params_dict_new

def estimate_angular_velocity_np(rot_seq, dRdt):
    # rot_seq: [T, 3, 3]
    # dRdt: [T, 3, 3]
    R = rot_seq
    RT = np.transpose(R, (0, -1, -2))
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)  # [B, T, ..., 3]
    return w

def get_logger(logdir):
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)