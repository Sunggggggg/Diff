import configargparse
import os
import random
import torch
import sys

# Diffusion utils
from utils.model_util import create_gaussian_diffusion
from diffusion import gaussian_diffusion
from diffusion.respace import SpacedDiffusion

# Network
from model.network import Network

# Train
from tensorboardX import SummaryWriter
from train.train_loop import TrainNetwork
from utils.other_utils import get_logger, save_config


arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
cfg_parser = configargparse.YAMLConfigFileParser
description = 'code'
group = configargparse.ArgParser(formatter_class=arg_formatter, config_file_parser_class=cfg_parser, 
                                 description=description, prog='')
group.add_argument("--expname", type=str)

######################## diffusion setups
group.add_argument("--diffusion_steps", default=100, type=int, help='diffusion time steps')
group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str, help="Noise schedule type")
group.add_argument("--timestep_respacing_eval", default='', type=str)  # if use ddim, set to 'ddimN', where N denotes ddim sampling steps
group.add_argument("--sigma_small", default='True', type=lambda x: x.lower() in ['true', '1'], help="Use smaller sigma values.")

####################### training setups
group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
group.add_argument('--debug', default='False', type=lambda x: x.lower() in ['true', '1'], help='')
group.add_argument("--max_infill_ratio", default=0.1, type=float, help="maximum occlusion ratio for traj infilling")
group.add_argument("--mask_prob", default=0.4, type=float, help="probability to apply occlusion mask for traj infilling")
group.add_argument("--start_infill_epoch", default=100000000000000000000, type=int, help="which epoch to start traj infilling")
group.add_argument("--save_dir", default='runs', type=str, help="Path to save checkpoints and results.")
group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
group.add_argument("--log_interval", default=25000, type=int)
group.add_argument("--save_interval", default=25000, type=int)
group.add_argument("--num_steps", default=1000000_000, type=int)
args = group.parse_args()

def main(args, writer, logdir, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ################ Logging
    

    ################ Loading dataset loader
    print("creating data loader...")

    ################ Create gaussian diff
    print("creating model and diffusion...")
    diffusion_train = create_gaussian_diffusion(args, gd=gaussian_diffusion,
                                                return_class=SpacedDiffusion,
                                                num_diffusion_timesteps=args.diffusion_steps,
                                                timestep_respacing='', device=device)
    
    diffusion_eval = create_gaussian_diffusion(args, gd=gaussian_diffusion,
                                                return_class=SpacedDiffusion,
                                                num_diffusion_timesteps=args.diffusion_steps,
                                                timestep_respacing=args.timestep_respacing_eval, device=device)

    model = Network(time_dim=32, mid_dim=512,
                    cond_dim=train_dataset.traj_feat_dim, traj_feat_dim=train_dataset.traj_feat_dim,
                    trajcontrol=args.trajcontrol,
                    device=device,
                    dataset=train_dataset,
                    repr_abs_only=args.repr_abs_only,
                    weight_loss_root_rec_repr=args.weight_loss_root_rec_repr,
                    weight_loss_root_smooth=args.weight_loss_root_smooth,
                    weight_loss_root_pos_global=args.weight_loss_root_pos_global,
                    weight_loss_root_vel_global=args.weight_loss_root_vel_global,
                    weight_loss_root_rot_vel_from_abs_traj=args.weight_loss_root_rot_vel_from_abs_traj,
                    weight_loss_root_smplx_rot_vel=args.weight_loss_root_smplx_rot_vel,
                    weight_loss_root_smplx_transl_vel=args.weight_loss_root_smplx_transl_vel,
                    weight_loss_root_rot_cos_smooth_from_abs_traj=args.weight_loss_root_rot_cos_smooth_from_abs_traj,
                    ).to(device)
    
    if args.load_pretrained_model:
        weights = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(weights)
        print('loaded checkpoint from {}'.format(args.pretrained_model_path))
    
    print("Training...")
    TrainNetwork(args, writer=writer, model=model,
                     diffusion_train=diffusion_train, diffusion_eval=diffusion_eval,
                     timestep_respacing_eval=args.timestep_respacing_eval,
                     start_infill_epoch=args.start_infill_epoch, max_infill_ratio=args.max_infill_ratio, mask_prob=args.mask_prob,
                     train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                     logdir=logdir, logger=logger, device=device
                     ).run_loop()


if __name__ == '__main__':
    logdir = os.path.join(args.save_dir, args.expname)
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()

    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    main(args, writer, logdir, logger)