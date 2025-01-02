# from GLAMR_data_loaders.amass_dataset import AMASSDataset

# AMASSDataset('trans_data', 'val', 81)
# import torch
# from model.network import Network
# B, T = 3, 16
# batch = {
#     'x_t' : torch.rand((B, T, 4)),
#     'cond' : torch.rand((B, T, 4)),
#     'time' : torch.rand((B))
# }
# model = Network()

# output = model(batch, batch['time'])

from Custom_data_loaders.amass_dataloader import DataloaderAMASS
DataloaderAMASS('trans_data', 'val', 81)