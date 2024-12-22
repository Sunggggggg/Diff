from torch.utils import data
from tqdm import tqdm
import glob
import smplx
from data_loaders.motion_representation import *
import pickle as pkl
from utils.other_utils import REPR_LIST, REPR_DIM_DICT

class DataloaderAMASS(data.Dataset):
    def __init__(self,
                 amass_root='',
                 spacing=1,):
        super().__init__()
        self.n_samples
    

    def create_body_repr(self, ):
        smpl_noise_dict = {}
        for i in tqdm(range(0, self.n_samples, self.spacing)) :
            source_data_joints = self.joints_clip_list[i][:, :self.joints_num, :]  # [T, 23, 3]
            source_data_smpl = self.smpl_clip_list[i]                               # [T, ]
            
            smpl_params_dict = {'global_orient': source_data_smpl[:, 0:3],
                                 'transl': source_data_smpl[:, 3:6],
                                 'betas': source_data_smpl[:, 6:16],
                                 'body_pose': source_data_smpl[:, 16:(16+63)],
                                 } 
            
            ######################################## canonicalize for GT sequence
            cano_positions, cano_smplx_params_dict = cano_seq_smplx(positions=source_data_joints,
                                                                    smpl_params_dict=smpl_params_dict,
                                                                    smpl_model=self.smpl_neutral, device=self.device)

        
        return