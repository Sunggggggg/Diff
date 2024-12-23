from torch.utils import data
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import glob
import smplx
from data_loaders.motion_representation import *
import pickle as pkl
from utils.other_utils import REPR_LIST, REPR_DIM_DICT

from configs import constants as _C


class DataloaderAMASS(data.Dataset):
    def __init__(self,
                 amass_root='',
                 spacing=1,):
        super().__init__()
        self.n_samples
        self.smpl_neutral = smplx.create(model_path=_C.BMODEL.FLDR, model_type='smpl', gender='neutral')
    

    def create_body_repr(self, ):
        smpl_noise_dict = {}
        for i in tqdm(range(0, self.n_samples, self.spacing)) :
            source_data_joints = self.joints_clip_list[i][:, :self.joints_num, :]   # [T, 23, 3]
            source_data_smpl = self.smpl_clip_list[i]                               # [T, ]
            
            smpl_params_dict = {'global_orient': source_data_smpl[:, 0:3],
                                 'transl': source_data_smpl[:, 3:6],
                                 'betas': source_data_smpl[:, 6:16],
                                 'body_pose': source_data_smpl[:, 16:(16+63)],
                                 } 
            
            ######################################## canonicalize for GT sequence
            cano_positions, cano_smpl_params_dict = cano_seq_smpl(positions=source_data_joints, smpl_params_dict=smpl_params_dict,
                                                                    smpl_model=self.smpl_neutral, device=self.device)
            
            ######################################## add noise to smpl params
            if self.input_noise and (not self.sep_noise):
                cano_smpl_params_dict_noisy = {}
                for param_name in ['transl', 'body_pose', 'betas', 'global_orient']:
                    # For Translation or Shape
                    if param_name == 'transl' or param_name == 'betas':
                        if self.load_noise:
                            noise_1 = self.loaded_smpl_noise_dict[param_name][i*self.spacing]
                        else:
                            noise_1 = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=cano_smpl_params_dict[param_name].shape)
                        cano_smpl_params_dict_noisy[param_name] = cano_smpl_params_dict[param_name] + noise_1
                        if param_name not in smpl_noise_dict.keys():
                            smpl_noise_dict[param_name] = []
                        smpl_noise_dict[param_name].append(noise_1)
                    
                    # Global orient
                    elif param_name == 'global_orient':
                        global_orient_mat = R.from_rotvec(cano_smpl_params_dict['global_orient'])  # [145, 3, 3]
                        global_orient_angle = global_orient_mat.as_euler('zxy', degrees=True)
                        if self.load_noise:
                            noise_global_rot = self.loaded_smpl_noise_dict[param_name][i*self.spacing]
                        else:
                            noise_global_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=global_orient_angle.shape)
                        global_orient_angle_noisy = global_orient_angle + noise_global_rot
                        cano_smpl_params_dict_noisy[param_name] = R.from_euler('zxy', global_orient_angle_noisy, degrees=True).as_rotvec()
                        if param_name not in smpl_noise_dict.keys():
                            smpl_noise_dict[param_name] = []
                        smpl_noise_dict[param_name].append(noise_global_rot)  #  [145, 3] in euler angle
                    
                    # Body pose
                    elif param_name == 'body_pose':
                        body_pose_mat = R.from_rotvec(cano_smpl_params_dict['body_pose'].reshape(-1, 3))
                        body_pose_angle = body_pose_mat.as_euler('zxy', degrees=True)  # [145*21, 3]
                        if self.load_noise:
                            noise_body_pose_rot = self.loaded_smpl_noise_dict[param_name][i*self.spacing].reshape(-1, 3)
                        else:
                            noise_body_pose_rot = np.random.normal(loc=0.0, scale=self.noise_std_params_dict[param_name], size=body_pose_angle.shape)
                        body_pose_angle_noisy = body_pose_angle + noise_body_pose_rot
                        cano_smpl_params_dict_noisy[param_name] = R.from_euler('zxy', body_pose_angle_noisy, degrees=True).as_rotvec().reshape(-1, 21, 3)
                        if param_name not in smpl_noise_dict.keys():
                            smpl_noise_dict[param_name] = []
                        smpl_noise_dict[param_name].append(noise_body_pose_rot.reshape(-1, 21, 3))  # [145, 21, 3]  in euler angle

                ### using FK to obtain noisy joint positions from noisy smpl params
                smpl_params_dict_noisy_torch = {}
                for key in cano_smpl_params_dict_noisy.keys():
                    smpl_params_dict_noisy_torch[key] = torch.FloatTensor(cano_smpl_params_dict_noisy[key]).to(self.device)
                bs = smpl_params_dict_noisy_torch['transl'].shape[0]
                # we do not consider face/hand details in RoHM
                smpl_params_dict_noisy_torch['jaw_pose'] = torch.zeros(bs, 3).to(self.device)
                smpl_params_dict_noisy_torch['leye_pose'] = torch.zeros(bs, 3).to(self.device)
                smpl_params_dict_noisy_torch['reye_pose'] = torch.zeros(bs, 3).to(self.device)
                smpl_params_dict_noisy_torch['left_hand_pose'] = torch.zeros(bs, 45).to(self.device)
                smpl_params_dict_noisy_torch['right_hand_pose'] = torch.zeros(bs, 45).to(self.device)
                smpl_params_dict_noisy_torch['expression'] = torch.zeros(bs, 10).to(self.device)
                cano_positions_noisy = self.smpl_neutral(**smpl_params_dict_noisy_torch).joints[:, 0:22].detach().cpu().numpy()  # [clip_len, 22, 3]

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
            for param_name in ['global_orient', 'transl', 'body_pose', 'betas']:
                self.smpl_params_list_dict[param_name].append(cano_smpl_params_dict[param_name])

            if self.input_noise and (not self.sep_noise):
                self.joints_noisy_list.append(cano_positions_noisy)
                for repr_name in REPR_LIST:
                    self.repr_list_dict_noisy[repr_name].append(repr_dict_noisy[repr_name])


            ######### FOR DEBUG: rec_ric_data should be same as cano_positions
            # repr_dict_torch = {}
            # for key in repr_dict.keys():
            #     repr_dict_torch[key] = torch.from_numpy(repr_dict[key]).unsqueeze(0).float().to(self.device)
            # rec_ric_data_clean = recover_from_repr_smpl(repr_dict_torch,
            #                                             recover_mode='smpl_params', smpl_model=self.smpl_neutral)  # [1, T-1, 22, 3]
            # rec_ric_data_clean = rec_ric_data_clean.detach().cpu().numpy()[0]  # [T-1, 22, 3]

        # ####################################### save smpl param noise
        # import pickle
        # for param_name in smpl_noise_dict.keys():
        #     smpl_noise_dict[param_name] = np.asarray(smpl_noise_dict[param_name])
        # pkl_path = 'eval_noise_smpl/smpl_noise_level_9.pkl'
        # with open(pkl_path, 'wb') as result_file:
        #     pickle.dump(smpl_noise_dict, result_file, protocol=2)
        # print('current smpl noise saved to{}.'.format(pkl_path))

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
            ######## save mean/std stats for the training data
            os.makedirs(save_dir) if not os.path.exists(save_dir) else None
            with open(os.path.join(save_dir, 'AMASS_mean.pkl'), 'wb') as result_file:
                pkl.dump(self.Mean_dict, result_file, protocol=2)
            with open(os.path.join(save_dir, 'AMASS_std.pkl'), 'wb') as result_file:
                pkl.dump(self.Std_dict, result_file, protocol=2)

        elif self.split == 'test':
            ######## load mean/std stats from the training data
            with open(os.path.join(save_dir, 'AMASS_mean.pkl'), 'rb') as f:
                self.Mean_dict = pkl.load(f)
            with open(os.path.join(save_dir, 'AMASS_std.pkl'), 'rb') as f:
                self.Std_dict = pkl.load(f)

        self.Mean = np.concatenate([self.Mean_dict[key] for key in self.Mean_dict.keys()], axis=-1)
        self.Std = np.concatenate([self.Std_dict[key] for key in self.Std_dict.keys()], axis=-1)