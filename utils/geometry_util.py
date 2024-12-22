import numpy as np
import torch
import torch.nn.functional as F

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

def rot6d_to_rotmat(x):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    # if rot6d_mode == 'prohmr':
    #     x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    # elif rot6d_mode == 'diffusion':
    x = x.reshape(-1, 3, 2)
    ### note: order for 6d feture items different between diffusion and prohmr code!!!
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)