from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
exp = Experiment(api_key="VeFWrxnaD0wAfQjDXoLy8gdiL",
                        project_name="3d-vae", workspace="axis-bit")

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.utils import data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

import pandas as pd
import numpy as np
from scipy import ndimage

from google_drive_downloader import GoogleDriveDownloader as gdd

from datasets import ShapeDataset
from networks.vae import VAE_CA_NET, Encoder, Decoder


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1280, help="size of the batches")
parser.add_argument("--data_path", type=str, default="./dataset/chair_table", help="size of the batches")
parser.add_argument("--root_path", type=str, default="./output/")
parser.add_argument('--z_size', type=int, default=256, help='beta for adam')
parser.add_argument('--cube_len', type=int, default=32 , help='size of noiz')
parser.add_argument('--channel', type=int, default=4, help='size of noiz')
parser.add_argument('--gpus', type=str, default="0", help='size of noiz')
parser.add_argument('--load', type=bool, default=False, help='size of noiz')



args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = 'cuda' if torch.cuda.is_available() else 'cpu'

decoder = Decoder()
ca = VAE_CA_NET(device)
encoder = Encoder()

# init
opt_enc = optim.Adam(encoder.parameters(), lr=0.0005)
opt_ca = optim.Adam(ca.parameters(), lr=0.0005)
opt_dec = optim.Adam(decoder.parameters(), lr=0.0005)


torch.backends.cudnn.benchmark=True

dataset = ShapeDataset(args.data_path)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True , num_workers=12 )

def safty_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

safty_makedirs("./vae/images")
safty_makedirs("./vae/models")


if args.load:
    state_dict = torch.load("./networks/pretrain/ca.pth", map_location=torch.device('cpu'))
    ca.load_state_dict(state_dict)
    state_dict = torch.load("./networks/pretrain/encoder.pth", map_location=torch.device('cpu'))
    encoder.load_state_dict(state_dict)
    state_dict = torch.load("./networks/pretrain/decoder.pth", map_location=torch.device('cpu'))
    decoder.load_state_dict(state_dict)

ca = torch.nn.DataParallel(ca)
encoder = torch.nn.DataParallel(encoder)
decoder = torch.nn.DataParallel(decoder)

decoder.to(device)
ca.to(device)
encoder.to(device)

ca.train()
encoder.train()
decoder.train()


def loss_function(recon_x, x, mu, logvar):
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    reconstruction_function = nn.MSELoss(reduction='sum')
    BCE = reconstruction_function(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


for i in range(args.epochs):
    print(f"\nEpoch: {i+1:d}")
    for idx,  shape in enumerate(loader):
        start_t = time.time()

        inputs = shape.to(device)
        inputs = inputs.view(-1, 4, args.cube_len, args.cube_len, args.cube_len)


        opt_enc.zero_grad()
        opt_ca.zero_grad()
        opt_dec.zero_grad()


        z, mean = encoder(inputs)
        z, mu, logvar = ca(z)
        recon_batch = decoder(z)

        loss = loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()

        opt_enc.step()
        opt_ca.step()
        opt_dec.step()

        end_t = time.time()
        if idx % 3 == 0:
            print('''[%d/%d][%d/%d] Loss: %.2f Time: %.2fs''' % (i, args.epochs, idx, len(loader), loss.item(), end_t - start_t))
            e_time = 250 * len(loader) * (end_t - start_t) / 60 / 60
            exp.log_metric('loss', loss.item(), step=idx, epoch=i)

    if i % 10 == 1:
        torch.save(encoder.module.state_dict(),'./vae/models/encoder%d.pth' % (i))
        torch.save(ca.module.state_dict(),'./vae/models/ca%d.pth' % (i))
        torch.save(decoder.module.state_dict(),'./vae/models/decoder%d.pth' % (i))

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        print("build_image: " + str(i))
        ax = fig.gca(projection='3d')
        ax.voxels(np.round(recon_batch[1][3].cpu().data.numpy()))
        exp.log_figure(figure_name=str(i), figure=plt, step=i)


        # np.save("./vae/images/0.npy", recon_batch[0].cpu().data.numpy())
        # np.save("./vae/images/1.npy", recon_batch[1].cpu().data.numpy())
        # np.save("./vae/images/2.npy", recon_batch[2].cpu().data.numpy())
        # np.save("./vae/images/3.npy", recon_batch[3].cpu().data.numpy())
        # np.save("./vae/images/4.npy", recon_batch[4].cpu().data.numpy())