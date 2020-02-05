
import os
import errno
import numpy as np
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def build_images(real_imgs, captions, ixtoword, fullpath):

    num = 4
    fig = plt.figure(figsize=(8, 4))
    captions = captions.detach().cpu()
    for i in range(num):
        sent= []
        for n in range(len(captions[0])):
            s_id = captions.numpy()[i][n]
            txt = ixtoword[int(s_id)]
            if(s_id != 0):
                sent.append(txt)


        ax = fig.add_subplot(num, 1, i + 1, projection='3d')
        ax = fig.gca(projection='3d')
        ax.voxels(np.round(real_imgs[i][3].detach().cpu().numpy()), edgecolor='b')
        ax.set_title("{}".format(' '.join(sent)))

    plt.savefig(fullpath+".png", bbox_inches='tight')

    return None

def build_npy(real_imgs, captions, ixtoword, fullpath):
    num = 8
    for i in range(num):
        np.save(fullpath+"_"+ str(i) + ".npy", real_imgs[i] )


def build_image(real_imgs, captions, ixtoword, step, exp):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    print("build_image: " + str(step))
    ax = fig.gca(projection='3d')
    ax.voxels(np.round(real_imgs[1][3].numpy()))
    exp.log_figure(figure_name=str(step), figure=plt, step=step)

def build_super_images(real_imgs, captions, cap_len, ixtoword, attn_maps, att_sze, exp, fullpath, step):
    num = 5
    for i in range(num):
        cap_ll = cap_len[i].numpy() + 1
        width = int(cap_ll * 14)
        fig = plt.figure(figsize=(width, 8), dpi=71, tight_layout=True, facecolor="w")
        plt.ioff()
        attn = attn_maps[i]
        real = real_imgs[i][3]
        ax = fig.add_subplot(1, cap_ll, 1, projection='3d')
        ax = fig.gca(projection='3d')
        for n in range(cap_ll - 1):
            s_id = captions.cpu().numpy()[i][n]
            if(s_id != 0):
                ax = fig.add_subplot(1, cap_ll, (n+1) + 1, projection='3d')
                ax = fig.gca(projection='3d')
                data = attn[0][n].add_(1).div_(2)
                data = data - data.min()
                data = data / (data.max() - data.min())

                data = data.cpu().detach().numpy()

                ax.voxels(np.round(data), edgecolor='k')
                txt = ixtoword[s_id]
                ax.set_title("{}".format(txt))

        name = str(i) + "_" + str(step)
        exp.log_figure(figure_name=name, figure=plt, step=step)
    return None