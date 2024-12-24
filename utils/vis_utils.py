import torch
import matplotlib.pyplot as plt

def trajectory_visual(transl):
    if isinstance(transl, torch.Tensor):
        transl = transl.detach().cpu().numpy()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(transl[:, 0], transl[:, 1], transl[:, 2])
    plt.savefig('traject.png')
    plt.close()