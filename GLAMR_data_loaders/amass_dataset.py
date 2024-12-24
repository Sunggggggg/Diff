import os
import numpy as np
import joblib
from collections import defaultdict
import torch
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import transforms
from GLAMR_data_loaders.augmentor import SequenceAugmentor
from GLAMR_data_loaders.utils import smpl_params_dim

class AMASSDataset(Dataset):
    def __init__(self, 
                 data_root,
                 split,
                 seqlen,
                 input_noise=False,
                 noise_std_smpl_global_rot=0.0,
                 noise_std_smpl_body_rot=0.0,
                 noise_std_smpl_trans=0.0,
                 noise_std_smpl_betas=0.0,
                
                 logdir='trans_data'):
        super().__init__()
        self.split = split
        self.seqlen = seqlen
        self.logdir = logdir
        self.input_noise = input_noise
        ########################################## Augmentation
        self.sequnceaugmentor = SequenceAugmentor(seqlen)
        self.noise_std_params_dict = {'global_orient': noise_std_smpl_global_rot,
                                      'transl': noise_std_smpl_trans,
                                      'body_pose': noise_std_smpl_body_rot,
                                      'betas': noise_std_smpl_betas,}
        
        ########################################## Read data
        self.coco_joints_clip_list, self.smpl_clip_list = [], []
        self.read_data(data_root)
        self.preprocess()

    def read_data(self, data_root='./trans_data'):
        split_data_root = os.path.join(data_root, self.split)
        data_name_list = os.listdir(split_data_root)
        for data_name in tqdm(data_name_list):
            data_path = os.path.join(split_data_root, data_name)
            db = joblib.load(data_path)
            N = len(db['kp3d'])
            if N >= self.seqlen :
                num_valid_clip = int(N / self.seqlen)
                for i in range(num_valid_clip):
                    root_pose = torch.from_numpy(db['root_pose'][:, np.newaxis][self.seqlen*i : self.seqlen * (i+1)])
                    body_pose = torch.from_numpy(db['body_pose'][self.seqlen*i : self.seqlen * (i+1)])
                    trans = torch.from_numpy(db['trans'][self.seqlen*i : self.seqlen * (i+1)])
                    shape = torch.from_numpy(db['shape'][self.seqlen*i : self.seqlen * (i+1)])
                    smpl_param = torch.cat([root_pose, trans, shape, body_pose], -1)

                    self.smpl_clip_list.append(smpl_param)

        self.n_samples = len(self.coco_joints_clip_list)
        print('[INFO] {} set: get {} sub clips in total.'.format(self.split, self.n_samples))

    def preprocess(self, ):
        self.input_dict_list, self.output_dict_list = [], []

        for i in tqdm(range(0, self.n_samples)):
            smpl_clip = self.smpl_clip_list[i].copy()
            root_pose, transl, shape, body_pose = smpl_clip[..., :3], smpl_clip[..., 3:6], smpl_clip[..., 6:16], smpl_clip[..., 16:]
            # pose = np.concatenate([root_pose[:, np.newaxis], body_pose], axis=-2)   # [T, 24, 3]
            
            smpl_params_dict_clean = {
                'global_orient': root_pose, # [T, 3]
                'transl': transl,           # [T, 3]
                'betas': shape,             # [T, 10]
                'body_pose': body_pose      # [T, 23, 3]
            }
            
            if self.input_noise :
                smpl_params_dict_noisy, smpl_noise_dict = defaultdict(list), defaultdict(list)
                for param_name in smpl_params_dict_clean.keys():
                    std = self.noise_std_params_dict[param_name]
                    # size = smpl_params_dict_clean[param_name].shape
                    noise = torch.normal(mean=0.0, std=std)
                    smpl_noise_dict[param_name].append(noise)
                    smpl_params_dict_noisy[param_name].append(smpl_params_dict_clean[param_name]+noise)
            else :
                smpl_params_dict_noisy = smpl_params_dict_clean

            self.input_dict_list.append(smpl_params_dict_noisy)
            self.output_dict_list.append(smpl_params_dict_clean)
        
        data_dict = {}
        for repr_name in ['global_orient', 'transl', 'betas', 'body_pose']:
            data_dict[repr_name] = np.asarray(self.output_dict_list[repr_name]) 
    
        save_dir = self.logdir
        if self.split == 'train' :
            self.mean_dict, self.std_dict = {}, {}
            for repr_name in ['global_orient', 'transl', 'betas', 'body_pose']:
                dim = smpl_params_dim[repr_name]
                
                self.mean_dict[repr_name] = smpl_params_dict_clean[repr_name].reshape(-1, dim).mean(axis=0).astype(np.float32)
                self.std_dict[repr_name] = smpl_params_dict_clean[repr_name].reshape(-1, dim).std(axis=0).astype(np.float32)

            os.makedirs(save_dir) if not os.path.exists(save_dir) else None
            with open(os.path.join(save_dir, 'AMASS_mean.pkl'), 'wb') as f:
                pkl.dump(self.mean_dict, f, protocol=2)
            with open(os.path.join(save_dir, 'AMASS_std.pkl'), 'wb') as f:
                pkl.dump(self.std_dict, f, protocol=2)
        
        elif self.split == 'test' :
            with open(os.path.join(save_dir, 'AMASS_mean.pkl'), 'rb') as f:
                self.mean_dict = pkl.load(f)
            with open(os.path.join(save_dir, 'AMASS_std.pkl'), 'rb') as f:
                self.std_dict = pkl.load(f)

        self.mean = np.concatenate([self.mean_dict[key] for key in self.mean_dict.keys()], axis=-1)
        self.std = np.concatenate([self.std_dict[key] for key in self.std_dict.keys()], axis=-1)

    def __len__(self) :
       return self.n_samples
    
    def __getitem__(self, idx):
        ######## Sampling data (dict type)
        input_dict, output_dict = self.input_dict_list[idx], self.output_dict_list[idx]
        input_aug_dict, l = self.sequnceaugmentor(input_dict)
        output_aug_dict, _ = self.sequnceaugmentor(output_dict, l=l)

        input_dict.update(input_aug_dict)
        output_dict.update(output_aug_dict)
        
        return input_dict, output_dict