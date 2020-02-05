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
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

import pandas as pd
import numpy as np
from scipy import ndimage

from networks.utility import GLU

class VAE_CA_NET(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device =  device
        self.z_size = 512

        self.fc_mu = nn.Linear(256, self.z_size)
        self.fc_logvar = nn.Linear(256, self.z_size)
        self.fc_rep = nn.Linear(self.z_size, 256)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def encode(self, h):
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        c_code, mu, logvar = self.encode(x)
        c_code = self.fc_rep(c_code)
        return c_code, mu, logvar

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnch1 = nn.Conv3d(4, 32, kernel_size=4, stride=2, padding=1)
        self.res11 = ResBlock(32)
        self.res12 = ResBlock(32)
        self.dnch2 = self.downsampling_module(32, 3)
        self.res21 = ResBlock(64)
        self.res22 = ResBlock(64)
        self.dnch3 = self.downsampling_module(64, 3)
        self.dnch4 = self.downsampling_module(128, 3)
        self.dnch5 = nn.AvgPool3d(2)

    def downsampling_module(self, input_channels, pooling_kenel):
        return nn.Sequential(
            nn.Conv3d(input_channels, input_channels*2, kernel_size=pooling_kenel, stride=2, padding=1),
            nn.BatchNorm3d(input_channels*2),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.dnch1(x)
        out = self.res11(out)
        out = self.res12(out)
        mean = out
        out = self.dnch2(out)
        out = self.res21(out)
        out = self.res22(out)
        out = self.dnch3(out)
        out = self.dnch4(out)
        out = self.dnch5(out)
        return out.view(-1, 256), mean

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upch1 = nn.ConvTranspose3d(32, 4, kernel_size=4, stride=2, padding=1)
        self.upch2 = self.upsampling_module(64, 4)
        self.upch3 = self.upsampling_module(128, 4)
        self.upch4 = self.upsampling_module(256, 4)
        self.upch5 = nn.Upsample(scale_factor=2, mode='nearest')

    def upsampling_module(self, input_channels, pooling_kenel):
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, input_channels//2, kernel_size=pooling_kenel, stride=2, padding=1),
            nn.BatchNorm3d(input_channels//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = x.view(-1, 256, 1, 1, 1)
        out = self.upch5(out)
        out = self.upch4(out)
        out = self.upch3(out)
        out = self.upch2(out)
        out = self.upch1(out)
        return torch.sigmoid(out)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm3d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm3d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out