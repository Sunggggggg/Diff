import torch
import torch.nn as nn
import einops
from model.head import *

class Network(nn.Module):
    def __init__(self,time_dim=32, cond_dim=4, mid_dim=256,
                 traj_feat_dim=4,
                 device=None, dataset=None,
                 repr_aB_only=False,
                 ####### trajcontrol setup
                 trajcontrol=False,
                 control_cond_dim=272,
                 ####### loss weights
                 weight_loss_root_rec_repr=0.0,
                 weight_loss_root_pos_global=0.0, weight_loss_root_vel_global=0.0,
                 weight_loss_root_rot_vel_from_aB_traj=0.0,
                 weight_loss_root_smplx_transl_vel=0.0, weight_loss_root_smplx_rot_vel=0.0,
                 weight_loss_root_smooth=0.0,
                 weight_loss_root_rot_cos_smooth_from_aB_traj=0.0,
                 ):
        super().__init__()

        self.traj_feat_dim = traj_feat_dim
        self.repr_aB_only = repr_aB_only

        self.weight_loss_root_rec_repr = weight_loss_root_rec_repr
        self.weight_loss_root_pos_global = weight_loss_root_pos_global
        self.weight_loss_root_vel_global = weight_loss_root_vel_global
        self.weight_loss_root_rot_vel_from_aB_traj = weight_loss_root_rot_vel_from_aB_traj
        self.weight_loss_root_smplx_transl_vel = weight_loss_root_smplx_transl_vel
        self.weight_loss_root_smplx_rot_vel = weight_loss_root_smplx_rot_vel
        self.weight_loss_root_smooth = weight_loss_root_smooth
        self.weight_loss_root_rot_cos_smooth_from_aB_traj = weight_loss_root_rot_cos_smooth_from_aB_traj

        self.mse_loss = nn.MSELoss(reduction='none').to(device)
        self.dataset = dataset
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        ############################ unet ###########################
        ########### encoder
        self.diff_enc1 = ResidualTemporalBlock(self.traj_feat_dim, mid_dim // 8, input_t=True, t_embed_dim=time_dim)
        self.diff_downsample1 = nn.Conv1d(mid_dim // 8, mid_dim // 8, kernel_size=3, stride=2, padding=1)

        self.diff_enc2 = ResidualTemporalBlock(mid_dim // 8, mid_dim // 4, input_t=True, t_embed_dim=time_dim)
        self.diff_downsample2 = nn.Conv1d(mid_dim // 4, mid_dim // 4, kernel_size=3, stride=2, padding=1)

        self.diff_enc3 = ResidualTemporalBlock(mid_dim // 4, mid_dim // 2, input_t=True, t_embed_dim=time_dim)
        self.diff_downsample3 = nn.Conv1d(mid_dim // 2, mid_dim // 2, kernel_size=3, stride=2, padding=1)

        self.diff_enc4 = ResidualTemporalBlock(mid_dim // 2, mid_dim, input_t=True, t_embed_dim=time_dim)
        self.diff_downsample4 = nn.Conv1d(mid_dim, mid_dim, kernel_size=3, stride=2, padding=1)

        ########### middle layers
        self.diff_mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, input_t=True, t_embed_dim=time_dim)
        self.diff_mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, input_t=True, t_embed_dim=time_dim)

        ########### decoder
        self.diff_upsample4 = Upsample1d(mid_dim)
        self.diff_dec4 = ResidualTemporalBlock(mid_dim*2, mid_dim // 2, input_t=True, t_embed_dim=time_dim)

        self.diff_upsample3 = Upsample1d(mid_dim // 2)
        self.diff_dec3 = ResidualTemporalBlock(mid_dim // 2 *2, mid_dim // 4, input_t=True, t_embed_dim=time_dim)

        self.diff_upsample2 = Upsample1d(mid_dim // 4)
        self.diff_dec2 = ResidualTemporalBlock(mid_dim // 4*2, mid_dim // 8, input_t=True, t_embed_dim=time_dim)

        self.diff_upsample1 = Upsample1d(mid_dim // 8)
        self.diff_dec1 = ResidualTemporalBlock(mid_dim // 8 * 2, 32, input_t=True, t_embed_dim=time_dim)

        self.diff_final_conv = nn.Sequential(
            Conv1dBlock(32, 32, kernel_size=5),
            nn.Conv1d(32, self.traj_feat_dim, 1),
        )

    def forward(self, batch, time):
        """
        Input:
            batch['x_t']: [B, T, traj_dim]
            time: [B] values in [0,timesteps)
        Output:
            x_diff: [B, T, traj_dim], reconstructed traj repr at timestep 0
        """
        x_diff = batch['x_t']       # [B, T, traj_dim]
        t = self.time_mlp(time)     # [B, 32]

        ############## U-Net
        x_diff = einops.rearrange(x_diff, 'B T D -> B D T')  # [B, traj_dim, T]
        h_diff = []

        ####### encoder
        x_diff = self.diff_enc1(x_diff, t)      # [B, mid_dim/8, T]
        h_diff.append(x_diff)
        x_diff = self.diff_downsample1(x_diff)  # [B, mid_dim/8 * 2, T/2]

        x_diff = self.diff_enc2(x_diff, t)      # [B, mid_dim/4, T/2]
        h_diff.append(x_diff)
        x_diff = self.diff_downsample2(x_diff)  # [B, mid_dim/4*2, T/4]

        x_diff = self.diff_enc3(x_diff, t)      # [B, mid_dim/2, T/4]
        h_diff.append(x_diff)
        x_diff = self.diff_downsample3(x_diff)  # [B, mid_dim/2*2, T/8]

        x_diff = self.diff_enc4(x_diff, t)      # [B, mid_dim, T/8]
        h_diff.append(x_diff)
        x_diff = self.diff_downsample4(x_diff)  # [B, mid_dim*2, T/16]

        ####### middle
        x_diff = self.diff_mid_block1(x_diff, t)  # [B, mid_dim, T/16]
        x_diff = self.diff_mid_block2(x_diff, t)  # [B, mid_dim, T/16]

        ####### decoder
        x_diff = self.diff_upsample4(x_diff)  # [B, mid_dim, T/8]
        x_diff = self.diff_dec4(torch.cat([x_diff, h_diff[-1]], dim=1), t)  # [B, mid_dim/2, T/8]

        x_diff = self.diff_upsample3(x_diff)  # [B, mid_dim/2, T/4]
        x_diff = self.diff_dec3(torch.cat([x_diff, h_diff[-2]], dim=1), t)  # [B, mid_dim/4, T/4]

        x_diff = self.diff_upsample2(x_diff)  # [B, mid_dim/4, T/2]
        x_diff = self.diff_dec2(torch.cat([x_diff, h_diff[-3]], dim=1), t)   # [B, mid_dim/8, T/2]

        x_diff = self.diff_upsample1(x_diff)  # [B, mid_dim/8, T]
        x_diff = self.diff_dec1(torch.cat([x_diff, h_diff[-4]], dim=1), t)  # [B, mid_dim/8, T]

        x_diff = self.diff_final_conv(x_diff)  # [B, traj_dim, T]
        x_diff = einops.rearrange(x_diff, 'b t h -> b h t')  # [B, T, traj_dim]
        return x_diff

    def compute_losses_with_smpl(self, batch, model_output, smplx_model=None):
        """
        Input:
            model_output: [B, T, motion_repr_dim]
            batch: contains gt data and noisy condition input
        Output:
            loss_dict: dictionary of loss items
        """
        loss_dict = {}

        ###################### loss on full motion repr
        if not self.repr_aB_only:
            full_repr_rec = torch.cat([model_output, batch['motion_repr_clean'][:, :, self.traj_feat_dim:]], dim=-1)
        else:
            full_repr_rec = batch['motion_repr_clean'].clone()
            full_repr_rec[..., 0] = model_output[..., 0]
            full_repr_rec[..., 2:4] = model_output[..., 1:3]
            full_repr_rec[..., 6] = model_output[..., 3]
            full_repr_rec[..., 7:13] = model_output[..., 4:10]
            full_repr_rec[..., 16:19] = model_output[..., 10:13]
        loss_rec_traj_repr_all = self.mse_loss(batch['motion_repr_clean'], full_repr_rec)  # [B, clip_len, xxx]

        ###################### loss on traj repr
        loss_dict['loss_repr_traj_root_rot_angle'] = loss_rec_traj_repr_all[:, :, 0].mean()
        loss_dict['loss_repr_traj_root_l_pos'] = loss_rec_traj_repr_all[:, :, 2:4].mean()
        loss_dict['loss_repr_traj_root_height'] = loss_rec_traj_repr_all[:, :, 6].mean()
        loss_dict['loss_repr_traj_smplx_rot_6d'] = loss_rec_traj_repr_all[:, :, 7:13].mean()
        loss_dict['loss_repr_traj_smplx_trans'] = loss_rec_traj_repr_all[:, :, 16:19].mean()
        if not self.repr_aB_only:
            loss_dict['loss_repr_traj_root_rot_angle_vel'] = loss_rec_traj_repr_all[:, :, 1].mean()
            loss_dict['loss_repr_traj_root_l_vel'] = loss_rec_traj_repr_all[:, :, 4:6].mean()
            loss_dict['loss_repr_traj_smplx_rot_vel'] = loss_rec_traj_repr_all[:, :, 13:16].mean()
            loss_dict['loss_repr_traj_smplx_trans_vel'] = loss_rec_traj_repr_all[:, :, 19:22].mean()
            loss_dict['loss_repr_traj'] = loss_rec_traj_repr_all[..., 0:self.traj_feat_dim].mean()
        else:
            loss_dict['loss_repr_traj'] = torch.cat([loss_rec_traj_repr_all[..., 0:1], loss_rec_traj_repr_all[..., 2:4],
                                                     loss_rec_traj_repr_all[..., 6:7], loss_rec_traj_repr_all[..., 7:13],
                                                     loss_rec_traj_repr_all[..., 16:19]], dim=-1).mean()

        ###################### loss on pelvis (root) joint location
        full_repr_clean = batch['motion_repr_clean'] * torch.from_numpy(self.dataset.Std).to(self.device) + torch.from_numpy(self.dataset.Mean).to(self.device)
        # reconstruct joint positions
        cur_total_dim = 0
        repr_dict_clean = {}
        for repr_name in REPR_LIST:
            repr_dict_clean[repr_name] = full_repr_clean[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
            cur_total_dim += REPR_DIM_DICT[repr_name]
        joint_pos_clean = recover_from_repr_smpl(repr_dict_clean, recover_mode='joint_aB_traj', smplx_model=smplx_model)
        root_pos_clean = joint_pos_clean[:, :, 0]

        full_repr_rec = full_repr_rec * torch.from_numpy(self.dataset.Std).to(self.device) + torch.from_numpy(self.dataset.Mean).to(self.device)
        # reconstruct joint positions
        cur_total_dim = 0
        repr_dict_rec = {}
        for repr_name in REPR_LIST:
            repr_dict_rec[repr_name] = full_repr_rec[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
            cur_total_dim += REPR_DIM_DICT[repr_name]
        ### reconstruct joint positions from: aBolute traj repr (joint-based), relative traj repr (joint-based), and smplx-based repr
        # Note: relative traj repr (joint-based) in repr_dict_rec is actually ground truth if self.repr_aB_only=True, and corresponding loss will be 0
        joint_pos_rec_from_aB_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_aB_traj', smplx_model=smplx_model)  # [B, clip_len, 22, 3]
        joint_pos_rec_from_rel_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_rel_traj', smplx_model=smplx_model)
        joint_pos_rec_from_smpl = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=smplx_model)
        root_pos_rec_from_aB_traj = joint_pos_rec_from_aB_traj[:, :, 0]
        root_pos_rec_from_rel_traj = joint_pos_rec_from_rel_traj[:, :, 0]
        root_pos_rec_from_smpl = joint_pos_rec_from_smpl[:, :, 0]

        loss_dict['loss_root_pos_global_from_aB_traj'] = self.mse_loss(root_pos_rec_from_aB_traj, root_pos_clean).mean()
        loss_dict['loss_root_pos_global_from_rel_traj'] = self.mse_loss(root_pos_rec_from_rel_traj, root_pos_clean).mean()
        loss_dict['loss_root_pos_global_from_smpl'] = self.mse_loss(root_pos_rec_from_smpl, root_pos_clean).mean()

        ###################### loss on pelvis (root) joint velocity
        root_vel_clean = root_pos_clean[:, 1:] - root_pos_clean[:, 0:-1]
        root_vel_rec_from_aB_traj = root_pos_rec_from_aB_traj[:, 1:] - root_pos_rec_from_aB_traj[:, 0:-1]
        root_vel_rec_from_rel_traj = root_pos_rec_from_rel_traj[:, 1:] - root_pos_rec_from_rel_traj[:, 0:-1]
        root_vel_rec_from_smpl = root_pos_rec_from_smpl[:, 1:] - root_pos_rec_from_smpl[:, 0:-1]
        loss_dict['loss_root_vel_global_from_aB_traj'] = self.mse_loss(root_vel_rec_from_aB_traj, root_vel_clean).mean()
        loss_dict['loss_root_vel_global_from_rel_traj'] = self.mse_loss(root_vel_rec_from_rel_traj, root_vel_clean).mean()
        loss_dict['loss_root_vel_global_from_smpl'] = self.mse_loss(root_vel_rec_from_smpl, root_vel_clean).mean()

        ###################### loss on smplx global_orient angular velocity
        B = joint_pos_clean.shape[0]
        global_orient_mat = rot6d_to_rotmat(repr_dict_rec['smplx_rot_6d'].reshape(-1, 6))  # [B*T, 3, 3]
        global_orient_mat = global_orient_mat.reshape(B, -1, 3, 3)
        dRdt = global_orient_mat[:, 1:] - global_orient_mat[:, 0:-1]  # [B, seq_len-1, 3, 3]
        smplx_rot_vel = estimate_angular_velocity(global_orient_mat[:, 0:-1], dRdt)  # [B, 143, 3]
        loss_dict['loss_root_smplx_rot_vel'] = self.mse_loss(smplx_rot_vel, repr_dict_clean['smplx_rot_vel'][:, 0:-1]).mean()

        ###################### loss on smplx global transl velocity
        smplx_transl_vel = repr_dict_rec['smplx_trans'][:, 1:] - repr_dict_rec['smplx_trans'][:, 0:-1]
        loss_dict['loss_root_smplx_transl_vel'] = self.mse_loss(smplx_transl_vel, repr_dict_clean['smplx_trans_vel'][:, 0:-1]).mean()

        ###################### pelvis position translational smoothness loss
        root_acc_rec_from_aB_traj = root_vel_rec_from_aB_traj[:, 1:] - root_vel_rec_from_aB_traj[:, 0:-1]
        root_acc_rec_from_rel_traj = root_vel_rec_from_rel_traj[:, 1:] - root_vel_rec_from_rel_traj[:, 0:-1]
        root_acc_rec_from_smpl = root_vel_rec_from_smpl[:, 1:] - root_vel_rec_from_smpl[:, 0:-1]
        loss_dict['loss_root_smooth_from_aB_traj'] = torch.mean(root_acc_rec_from_aB_traj ** 2)
        loss_dict['loss_root_smooth_from_rel_traj'] = torch.mean(root_acc_rec_from_rel_traj ** 2)
        loss_dict['loss_root_smooth_from_smpl'] = torch.mean(root_acc_rec_from_smpl ** 2)

        ###################### pelvis rotation velocity and smoothness loss
        # compute on cosine values: continuous, no jump
        root_rot_cos_vel_clean = torch.cos(repr_dict_clean['root_rot_angle'][:, 1:] * 2) - torch.cos(repr_dict_clean['root_rot_angle'][:, 0:-1] * 2)
        root_rot_cos_vel_rec = torch.cos(repr_dict_rec['root_rot_angle'][:, 1:] * 2) - torch.cos(repr_dict_rec['root_rot_angle'][:, 0:-1] * 2)
        loss_dict['loss_root_rot_cos_vel_from_aB_traj'] = self.mse_loss(root_rot_cos_vel_clean, root_rot_cos_vel_rec).mean()

        root_rot_cos_acc_rec = root_rot_cos_vel_rec[:, 1:] - root_rot_cos_vel_rec[:, 0:-1]
        loss_dict['loss_root_rot_cos_smooth_from_aB_traj'] = torch.mean(root_rot_cos_acc_rec ** 2)

        if self.repr_aB_only:
            loss_dict['loss_root_pos_global_from_rel_traj'] = torch.tensor(0.0).to(self.device)
            loss_dict['loss_root_vel_global_from_rel_traj'] = torch.tensor(0.0).to(self.device)
            loss_dict['loss_root_smooth_from_rel_traj'] = torch.tensor(0.0).to(self.device)


        loss_dict["loss"] = self.weight_loss_root_rec_repr * loss_dict['loss_repr_traj'] + \
                            self.weight_loss_root_pos_global * (loss_dict['loss_root_pos_global_from_aB_traj'] + loss_dict['loss_root_pos_global_from_rel_traj'] + loss_dict['loss_root_pos_global_from_smpl']) + \
                            self.weight_loss_root_vel_global * (loss_dict['loss_root_vel_global_from_aB_traj'] + loss_dict['loss_root_vel_global_from_rel_traj'] + loss_dict['loss_root_vel_global_from_smpl']) + \
                            self.weight_loss_root_rot_vel_from_aB_traj * loss_dict['loss_root_rot_cos_vel_from_aB_traj'] + \
                            self.weight_loss_root_smplx_transl_vel * loss_dict['loss_root_smplx_transl_vel'] + \
                            self.weight_loss_root_smplx_rot_vel * loss_dict['loss_root_smplx_rot_vel'] + \
                            self.weight_loss_root_smooth * (loss_dict['loss_root_smooth_from_aB_traj'] + loss_dict['loss_root_smooth_from_rel_traj'] + loss_dict['loss_root_smooth_from_smpl']) + \
                            self.weight_loss_root_rot_cos_smooth_from_aB_traj * loss_dict['loss_root_rot_cos_smooth_from_aB_traj']
        return loss_dict
    
if __name__ == '__main__':
    B, T = 3, 16
    batch = {
        'x_t' : torch.rand((B, T, 4)),
        'cond' : torch.rand((B, T, 4)),
        'time' : torch.rand((B))
    }
    model = Network()

    output = model(batch)
    