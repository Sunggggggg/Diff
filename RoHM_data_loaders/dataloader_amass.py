import os
import numpy as np
import torch

from collections import defaultdict
from torch.utils import data
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import joblib
import smplx
import pickle as pkl
from utils.other_utils import REPR_LIST, REPR_DIM_DICT
from configs import constants as _C
from .motion_representation import *

def make_list_dict(_dict, keys) :
    for key in keys:
        _dict[key] = []
    return _dict

class DataloaderAMASS(data.Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 spacing=1,
                 repr_abs_only=False,
                 input_noise=False,
                 sep_noise=False,
                 noise_std_joint=0.0,
                 noise_std_smpl_global_rot=0.0,
                 noise_std_smpl_body_rot=0.0,
                 noise_std_smpl_trans=0.0,
                 noise_std_smpl_betas=0.0,
                 load_noise=False,
                 loaded_smplx_noise_dict=None,
                 task='traj',
                 seqlen=150, joints_num=22,
                 logdir=None, device='cpu'):
        super().__init__()
        self.n_samples
        self.seqlen = seqlen
        self.split = split
        self.smpl_neutral = smplx.create(model_path=_C.BMODEL.FLDR, model_type='smpl', gender='neutral')

        ########################################## configs about how to add noise
        self.input_noise = input_noise
        self.sep_noise = sep_noise  # add different noise to joint-based repr, and smplx-based repr, separately
        self.noise_std_joint = noise_std_joint  # set if sep_noise=True
        self.noise_std_params_dict = {'global_orient': noise_std_smpl_global_rot,
                                      'transl': noise_std_smpl_trans,
                                      'body_pose': noise_std_smpl_body_rot,
                                      'betas': noise_std_smpl_betas,}
        
        ########################################## Data block
        self.joints_clean_list, self.joints_noisy_list, self.joints_clip_list, self.smpl_clip_list = [], [], [], []
        self.repr_list_dict, self.repr_list_dict_noisy, self.smpl_params_list_dict = {}, {}, {}
        
        self.repr_list_dict = make_list_dict(self.repr_list_dict, keys=REPR_LIST)
        self.repr_list_dict_noisy = make_list_dict(self.repr_list_dict_noisy, keys=REPR_LIST)
        self.smpl_params_list_dict = make_list_dict(self.smpl_params_list_dict, keys=['global_orient', 'transl', 'body_pose', 'betas'])

        ########################################## Read data
        self.read_data(data_root)
        self.create_body_repr()
        
    def read_data(self, data_root='./trans_data'):
        data_name_list = os.listdir(data_root)
        for data_name in tqdm(data_name_list):
            data_path = os.path.join(data_root, self.split, data_name)
            db = joblib.load(data_path)
            N = len(db['kp3d'])
            if N >= self.seqlen :
                num_valid_clip = int(N / self.seqlen)
                for i in range(num_valid_clip):
                    kp3d = db['kp3d'][self.seqlen*i : self.seqlen * (i+1)]
                    root_pose = db['root_pose']
                    body_pose = db['body_pose']
                    trans = db['trans']
                    shape = db['shape']
                    smpl_param = np.concatenate([root_pose, trans, shape, body_pose], axis=-1)
                    
                    self.joints_clip_list.append(kp3d)
                    self.smpl_clip_list.append(smpl_param)
            else :
                continue

        self.n_samples = len(self.joints_clip_list)
        print('[INFO] {} set: get {} sub clips in total.'.format(self.split, self.n_samples))

    def create_body_repr(self, ):
        smpl_noise_dict = defaultdict(list)
        for i in tqdm(range(0, self.n_samples, self.spacing)) :
            source_data_joints = self.joints_clip_list[i][:, :self.joints_num, :]   # [T, 23, 3]
            source_data_smpl = self.smpl_clip_list[i]                               

            smpl_params_dict = {'global_orient': source_data_smpl[:, 0:3],  # [T, 3]
                                 'transl': source_data_smpl[:, 3:6],        # [T, 3]
                                 'betas': source_data_smpl[:, 6:16],        # [T, 10]
                                 'body_pose': source_data_smpl[:, 16:],     # [T, 23, 3]
                                 } 
            
            ######################################## canonicalize for GT sequence
            cano_positions, cano_smpl_params_dict = cano_seq_smpl(positions=source_data_joints, smpl_params_dict=smpl_params_dict,
                                                                    smpl_model=self.smpl_neutral, device=self.device)
            
            ######################################## add noise to smpl params
            if self.input_noise and (not self.sep_noise):
                cano_smpl_params_dict_noisy = {}
                for param_name in ['transl', 'body_pose', 'betas', 'global_orient']:
                    ###################  For Translation or Shape
                    if param_name == 'transl' or param_name == 'betas':
                        if self.load_noise:
                            noise_1 = self.loaded_smpl_noise_dict[param_name][i*self.spacing]
                        else:
                            noise_1 = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=cano_smpl_params_dict[param_name].shape)
                        cano_smpl_params_dict_noisy[param_name] = cano_smpl_params_dict[param_name] + noise_1
                        smpl_noise_dict[param_name].append(noise_1)
                    ########################################
                    
                    ################### Global orient
                    elif param_name == 'global_orient':
                        global_orient_mat = R.from_rotvec(cano_smpl_params_dict['global_orient'])   # [T, 3] (Axis-angle)
                        global_orient_angle = global_orient_mat.as_euler('zxy', degrees=True)       # [T, 3] (degree)
                        if self.load_noise:
                            noise_global_rot = self.loaded_smpl_noise_dict[param_name][i*self.spacing]
                        else:
                            noise_global_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=global_orient_angle.shape)
                        global_orient_angle_noisy = global_orient_angle + noise_global_rot
                        cano_smpl_params_dict_noisy[param_name] = R.from_euler('zxy', global_orient_angle_noisy, degrees=True).as_rotvec()
                        smpl_noise_dict[param_name].append(noise_global_rot)
                    ########################################
                    
                    # Body pose
                    elif param_name == 'body_pose':
                        body_pose_mat = R.from_rotvec(cano_smpl_params_dict['body_pose'].reshape(-1, 3))
                        body_pose_angle = body_pose_mat.as_euler('zxy', degrees=True)
                        if self.load_noise:
                            noise_body_pose_rot = self.loaded_smpl_noise_dict[param_name][i*self.spacing].reshape(-1, 3)
                        else:
                            noise_body_pose_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=body_pose_angle.shape)
                        body_pose_angle_noisy = body_pose_angle + noise_body_pose_rot
                        cano_smpl_params_dict_noisy[param_name] = R.from_euler('zxy', body_pose_angle_noisy, degrees=True).as_rotvec().reshape(-1, self.joints_num, 3)
                        smpl_noise_dict[param_name].append(noise_body_pose_rot.reshape(-1, self.joints_num, 3))  # [145, self.joints_num, 3]  in euler angle
                    ########################################

                ### using FK to obtain noisy joint positions from noisy smpl params
                smpl_params_dict_noisy_torch = {}
                for key in cano_smpl_params_dict_noisy.keys():
                    smpl_params_dict_noisy_torch[key] = torch.FloatTensor(cano_smpl_params_dict_noisy[key]).to(self.device)
                cano_positions_noisy = self.smpl_neutral(**smpl_params_dict_noisy_torch).joints[:, :23].detach().cpu().numpy()  # [clip_len, 22, 3]
                del smpl_params_dict_noisy_torch

            ######################################## create motion representation
            repr_dict = get_repr_smpl(positions=cano_positions,
                                       smpl_params_dict=cano_smpl_params_dict,
                                       feet_vel_thre=5e-5)  # a dict of reprs
            if self.input_noise and (not self.sep_noise):
                repr_dict_noisy = get_repr_smpl(positions=cano_positions_noisy,
                                                 smpl_params_dict=cano_smpl_params_dict_noisy,
                                                 feet_vel_thre=5e-5)  # a dict of reprs

            ############### clean data repr gt
            self.joints_clean_list.append(cano_positions)
            for repr_name in REPR_LIST:
                self.repr_list_dict[repr_name].append(repr_dict[repr_name])
            for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:    # GT
                self.smpl_params_list_dict[param_name].append(cano_smpl_params_dict[param_name])

            if self.input_noise and (not self.sep_noise):
                self.joints_noisy_list.append(cano_positions_noisy)
                for repr_name in REPR_LIST:
                    self.repr_list_dict_noisy[repr_name].append(repr_dict_noisy[repr_name])

        #######################################  get mean/std for dataset
        save_dir = self.logdir
        for repr_name in REPR_LIST:
            self.repr_list_dict[repr_name] = np.asarray(self.repr_list_dict[repr_name])  # each item: [N, T-1, d]
        
        if self.split == 'train':
            self.Mean_dict = {}
            self.Std_dict = {}
            for repr_name in REPR_LIST:
                self.Mean_dict[repr_name] = self.repr_list_dict[repr_name].reshape(-1, REPR_DIM_DICT[repr_name]).mean(axis=0).astype(np.float32)
                if repr_name == 'foot_contact':
                    self.Mean_dict[repr_name][...] = 0.0
                self.Std_dict[repr_name] = self.repr_list_dict[repr_name].reshape(-1, REPR_DIM_DICT[repr_name]).std(axis=0).astype(np.float32)
                # do not normalize for smpl beta (already in a normal distribution) and foot contact labels (0/1 label)
                if repr_name != 'smpl_betas' and repr_name != 'foot_contact':
                    self.Std_dict[repr_name][...] = self.Std_dict[repr_name].mean() / 1.0
                elif repr_name == 'foot_contact':
                    self.Std_dict[repr_name][...] = 1.0
            os.makedirs(save_dir) if not os.path.exists(save_dir) else None
            with open(os.path.join(save_dir, 'AMASS_mean.pkl'), 'wb') as result_file:
                pkl.dump(self.Mean_dict, result_file, protocol=2)
            with open(os.path.join(save_dir, 'AMASS_std.pkl'), 'wb') as result_file:
                pkl.dump(self.Std_dict, result_file, protocol=2)

        elif self.split == 'test':
            with open(os.path.join(save_dir, 'AMASS_mean.pkl'), 'rb') as f:
                self.Mean_dict = pkl.load(f)
            with open(os.path.join(save_dir, 'AMASS_std.pkl'), 'rb') as f:
                self.Std_dict = pkl.load(f)

        self.Mean = np.concatenate([self.Mean_dict[key] for key in self.Mean_dict.keys()], axis=-1)
        self.Std = np.concatenate([self.Std_dict[key] for key in self.Std_dict.keys()], axis=-1)

    def __len__(self):
        return self.n_samples // self.spacing

    def __getitem__(self, index):
        positions_clean = self.joints_clean_list[index]
        repr_dict_clean = {}
        for repr_name in REPR_LIST:
            repr_dict_clean[repr_name] = self.repr_list_dict[repr_name][index] 

        ####################################### add noise
        if self.input_noise:
            if self.sep_noise:
                smpl_params_dict_clean, smpl_params_dict_noisy = {}, {}
                for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
                    smpl_params_dict_clean[param_name] = self.smpl_params_list_dict[param_name][index]
                    
                    noise = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=smpl_params_dict_clean[param_name].shape)
                    smpl_params_dict_noisy[param_name] = smpl_params_dict_clean[param_name] + noise

                noise = np.random.normal(loc=0.0, scale=self.noise_std_joint, size=positions_clean.shape)
                positions_noisy = positions_clean + noise 
                positions_noisy = positions_noisy.astype(np.float32)
                
                repr_dict_noisy = get_repr_smpl(positions=positions_noisy, smpl_params_dict=smpl_params_dict_noisy,)
            else:
                positions_noisy = self.joints_noisy_list[index]
                repr_dict_noisy = {}
                for repr_name in REPR_LIST:
                    repr_dict_noisy[repr_name] = self.repr_list_dict_noisy[repr_name][index]

        item_dict = {}
        item_dict['motion_repr_clean'] = np.concatenate([repr_dict_clean[key] for key in REPR_LIST], axis=-1)
        if self.input_noise:
            item_dict['noisy_joints'] = positions_noisy
            item_dict['motion_repr_noisy'] = np.concatenate([repr_dict_noisy[key] for key in REPR_LIST], axis=-1)
        else:
            item_dict['motion_repr_noisy'] = item_dict['motion_repr_clean'].copy()

        item_dict['motion_repr_clean'] = ((item_dict['motion_repr_clean'] - self.Mean) / self.Std).astype(np.float32)
        item_dict['motion_repr_noisy'] = ((item_dict['motion_repr_noisy'] - self.Mean) / self.Std).astype(np.float32)

        if not self.repr_abs_only:
            noisy_traj = item_dict['motion_repr_noisy'][:, 0:self.traj_feat_dim]
        else:
            temp = item_dict['motion_repr_noisy']
            noisy_traj = np.concatenate([temp[..., [0]], temp[..., 2:4], temp[..., [6]], temp[..., 7:13], temp[..., 16:19]], axis=-1)  # [144, 13]
        item_dict['cond'] = noisy_traj

        return item_dict