
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from random import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random

from miscc.losses import sent_loss, words_loss
from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

from datasets import Dataset
from networks.vae import VAE_CA_NET, Encoder


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Talk2Shape network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfg/32_chair_table_local.yml', type=str)
    parser.add_argument('--size', dest='size', default=256, type=int)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def visualize_zs(zs, shapes, keys, labels):
  fig, ax = plt.subplots(figsize=(96,64))

  colors = ["red", "green"]
  ls = ["table","chair"]
  points = TSNE(n_components=2, random_state=0).fit_transform(zs)

  for i , (p, s, k, l) in enumerate(zip(points, shapes, keys, labels)):
    print(i)


    img = plt.imread("./tools/shape_captcha/images/"+ str(k) +".png", format='png')
    imb0 = OffsetImage(img, zoom=0.4)
    imb0.image.axes = ax
    plt.tick_params(labelsize=40)
    plt.grid(which='major',color='black',linestyle='-')
    ab0  = AnnotationBbox(imb0, p, xybox=(0, 0), xycoords="data", boxcoords="offset points", pad=0.1, bboxprops=dict(edgecolor=colors[l]))
    ax.add_artist(ab0)

    plt.scatter(p[0], p[1], s=100, marker="${}$".format(ls[l]), c=colors[l])
  plt.legend(ls, fontsize=64)
  plt.savefig('./plts/vae.png')



if __name__ == "__main__":

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID

    if cfg.RANDOM_SEED:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)

    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)

    data_path = cfg.DATA_DIR + cfg.DATASET_NAME
    dataset = Dataset(data_path, cfg.SPLIT, cfg.CATEGORY, cfg.CHANNEL)

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.size, drop_last=True,
        shuffle=False, num_workers=int(cfg.WORKERS))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder()
    ca = VAE_CA_NET(device)

    encoder.load_state_dict(torch.load("./networks/pretrain/encoder.pth", map_location=device))
    ca.load_state_dict(torch.load("./networks/pretrain/ca.pth", map_location=device))

    for step, data in enumerate(dataloader):
        shape, cap, cap_len, cls_id, key = data
        sp = shape.to(device)
        h, mean = encoder(sp)
        c_code, mu, logvar = ca(h)

        sent = c_code.cpu().detach().numpy()
        visualize_zs(sent, sp, key, cls_id)
        break
