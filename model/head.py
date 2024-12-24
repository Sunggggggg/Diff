import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

##################################################################
######################### Basic block ############################
##################################################################
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    
##################################################################
######################### Diff. utils block ######################
##################################################################
class ResidualTemporalBlock(nn.Module):
    """
    This class is copied from https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py
    """
    def __init__(self, inp_channels=4, out_channels=64, input_t=False, t_embed_dim=32, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.input_t = input_t
        if input_t:
            self.time_mlp = nn.Sequential(
                nn.Mish(),
                nn.Linear(t_embed_dim, out_channels),
                Rearrange('batch t -> batch t 1'),
            )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        if self.input_t:
            out = out + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)
    
##################################################################
######################### Temp. block ############################
##################################################################
class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hid_dim, n_layers, bidirectional=False, batch_first=True, dropout=0.3)
        self.output_proj = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        """
        x : [B, T, dim]
        """
        T = x.shape[1]
        h0 = torch.zeros_like(x[:, :1])

        context_feat_list = []
        for t in range(T):
            xout, h0 = self.rnn(x[:, [t]], h0)
            context_feat_list.append(self.output_proj(xout))

        context_feat = torch.stack(context_feat_list, dim=1)    # [B, T, dim]
        return context_feat