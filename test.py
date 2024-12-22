import torch
from model.network import Network

B, T = 3, 16
batch = {
    'x_t' : torch.rand((B, T, 4)),
    'cond' : torch.rand((B, T, 4)),
}
time = torch.rand((B))

model = Network()
output = model(batch, time)

print(output.shape)