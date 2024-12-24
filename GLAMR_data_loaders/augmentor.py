import torch
import torch.nn.functional as F
from utils import transforms

class SequenceAugmentor:
    l_factor = 1.5
    def __init__(self, l_default):
        self.l_default = l_default

    def __call__(self, data_dict, l=None):
        """
        data_dict
            'key1' : [T, dim]
            'key2' : [T, J, dim]
            ...
        """
        output_dict = {}
        if l is None :
            l = torch.randint(low=int(self.l_default / self.l_factor), high=int(self.l_default * self.l_factor), size=(1, ))
        
        for k, v in data_dict.items():
            if k == 'pose' :
                pose = transforms.matrix_to_rotation_6d(v)
                resampled_pose = F.interpolate(
                    pose[:l].permute(1, 2, 0), self.l_default, mode='linear', align_corners=True
                ).permute(2, 0, 1)
                resampled_pose = transforms.rotation_6d_to_matrix(resampled_pose)
                output_dict[k] = resampled_pose
            
            elif k == 'transl' :
                transl = v.unsqueeze(1)
                resampled_transl = F.interpolate(
                    transl[:l].permute(1, 2, 0), self.l_default, mode='linear', align_corners=True
                ).squeeze(0).T
                output_dict[k] = resampled_transl
            
            elif k == 'kp3d' :
                resampled_kp3d = F.interpolate(
                    v[:l].permute(1, 2, 0), self.l_default, mode='linear', align_corners=True
                ).permute(2, 0, 1)
                output_dict[k] = resampled_kp3d
            
            else :
                continue
        
        return output_dict, l