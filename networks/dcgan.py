import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.utils.spectral_norm as SpectralNorm
from miscc.config import cfg

from GlobalAttention import GlobalAttentionGeneral as ATT_NET
from networks.utility import GLU, conv3x3


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.z_dim = cfg.GAN.Z_DIM
        self.fc = nn.Linear(self.t_dim, self.z_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

class G_NET(nn.Module):
    def __init__(self, channel):
        super(G_NET, self).__init__()

        self.cube_len = 32
        self.z_size = 128
        self.channel = channel


        self.l1 = nn.Sequential(
            nn.ConvTranspose3d(self.z_size  , self.cube_len*8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.cube_len*8),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.cube_len*4),
            nn.ReLU(inplace=True)
        )

        self.l3 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.cube_len*2),
            nn.ReLU(inplace=True)
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len*2, self.cube_len , kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.cube_len),
            nn.ReLU(inplace=True)
        )
        self.l5 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len  , self.channel  , kernel_size = 4, stride = 2, padding = 1),
            nn.Sigmoid(),
        )
        self.ca_net = CA_NET()

    def forward(self, noise, sent_emb):

        c_code, mu, logvar = self.ca_net(sent_emb)
        # c_z_code = torch.cat((c_code, noise), 1)
        print(c_code[0:5])
        x = c_code.view(-1, 128, 1, 1, 1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x, mu, logvar

class D_NET(nn.Module):

    def __init__(self, channel):
        super(D_NET, self).__init__()

        self.cube_len = 32
        self.ef_dim = 256
        self.channel = channel

        self.l1 = nn.Sequential(
            SpectralNorm(nn.Conv3d( self.channel, self.cube_len, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(),
        )

        self.l2 = nn.Sequential(
            SpectralNorm(nn.Conv3d( self.cube_len , self.cube_len * 2, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(),
        )

        self.l3 = nn.Sequential(
            SpectralNorm(nn.Conv3d( self.cube_len *2 , self.cube_len * 4, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(),
        )

        self.l4 = nn.Sequential(
            SpectralNorm(nn.Conv3d( self.cube_len *4 , self.cube_len * 8, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(),
        )

        self.l5 = nn.Sequential(
            SpectralNorm(nn.Conv3d( self.cube_len * 16, self.cube_len * 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
        )

        self.l6 = nn.Sequential(
            nn.Conv3d( self.cube_len * 16, 1, kernel_size=4, stride=2, padding=1),
        )


    def forward(self, x, sent_emb):
        x = x.view(-1, self.channel, 32, 32, 32)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        #outlogits
        c_code = sent_emb.view(-1, self.ef_dim, 1, 1, 1)
        c_code = c_code.repeat(1, 1, 2, 2, 2)
        h_c_code = torch.cat((x, c_code), 1)

        x = self.l5(h_c_code)
        x = self.l6(x)

        x = x.view(-1)
        return x


class G_NET_NEXT(nn.Module):
    def __init__(self, channel):
        super(G_NET_NEXT, self).__init__()
        self.gf_dim = 256
        self.ef_dim = 256
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(3):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.att = ATT_NET(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 2)

    def forward(self, h_code, c_code, word_embs):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        # self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)

        return out_code, att