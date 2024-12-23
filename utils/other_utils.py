import copy
import os
import json
import logging
import datetime
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

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