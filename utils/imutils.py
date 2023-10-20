import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt 

def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16),:]

def denormalize_img(imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:,0,:,:] = imgs[:,0,:,:] * std[0] + mean[0]
    _imgs[:,1,:,:] = imgs[:,1,:,:] * std[1] + mean[1]
    _imgs[:,2,:,:] = imgs[:,2,:,:] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs

def denormalize_img2(imgs=None):
    #_imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs)

    return imgs / 255.0

def minmax_norm(x):
    for i in range(x.shape[0]):
        x[i,...] = x[i,...] - x[i,...].min()
        x[i,...] = x[i,...] / x[i,...].max()
    return x

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


# def tensorboard_attn(attns=None, size=[224, 224], n_pix=0, n_row=4):
#     n = len(attns)
#     imgs = []
#     for idx, attn in enumerate(attns):
#
#         b, hw, _ = attn.shape
#         h = w = int(np.sqrt(hw))
#
#         attn_ = attn.clone()  # - attn.min()
#         # attn_ = attn_ / attn_.max()
#         _n_pix = int(h * n_pix) * (w + 1)
#         attn_ = attn_[:, _n_pix, :].reshape(b, 1, h, w)
#
#         attn_ = F.interpolate(attn_, size=size, mode='bilinear', align_corners=True)
#
#         attn_ = attn_.cpu()[:, 0, :, :]
#
#         def minmax_norm(x):
#             for i in range(x.shape[0]):
#                 x[i, ...] = x[i, ...] - x[i, ...].min()
#                 x[i, ...] = x[i, ...] / x[i, ...].max()
#             return x
#
#         attn_ = minmax_norm(attn_)
#
#         attn_heatmap = plt.get_cmap('viridis')(attn_.numpy())[:, :, :, 0:3] * 255
#         attn_heatmap = torch.from_numpy(attn_heatmap).permute([0, 3, 2, 1])
#         imgs.append(attn_heatmap)
#     attn_img = torch.cat(imgs, dim=0)
#
#     grid_attn = torchvision.utils.make_grid(tensor=attn_img.type(torch.uint8), nrow=n_row).permute(0, 2, 1)
#
#     return grid_attn