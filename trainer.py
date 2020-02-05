import os
import time
import sys
import numpy as np
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


from miscc.losses import HingeLoss, KL_loss
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_npy, build_image
from networks.dcgan import G_NET, D_NET, G_NET_NEXT
from networks.rnn_cnn import RNN_ENCODER, CNN_ENCODER
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, stage=0):
      if cfg.SPLIT == 'train':
          self.model_dir = os.path.join(output_dir, 'Model')
          self.image_dir = os.path.join(output_dir, 'Image')
          self.shape_dir = os.path.join(output_dir, 'Shape')
          mkdir_p(self.model_dir)
          mkdir_p(self.image_dir)
          mkdir_p(self.shape_dir)

      cudnn.benchmark = True

      self.batch_size = cfg.TRAIN.BATCH_SIZE
      self.config_name = cfg.CONFIG_NAME
      self.max_epoch = cfg.TRAIN.MAX_EPOCH
      self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
      self.nz = cfg.GAN.Z_DIM

      self.n_words = n_words
      self.ixtoword = ixtoword
      self.data_loader = data_loader
      self.num_batches = len(self.data_loader)
      self.stage = stage

      self.start_epoch = 0
      self.exp = Experiment(api_key="VeFWrxnaD0wAfQjDXoLy8gdiL", project_name=self.config_name, workspace="axis-bit")
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      self.gan_loss = HingeLoss(self.batch_size, self.device)


    def build_models(self):
        netG = G_NET(cfg.CHANNEL).to(self.device)
        netD = D_NET(cfg.CHANNEL).to(self.device)


        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load("./networks/pretrain/text_encoder.pth", map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        text_encoder.eval()
        text_encoder.to(self.device)

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load("./networks/pretrain/image_encoder.pth", map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        image_encoder.eval()
        image_encoder.to(self.device)

        optG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.99))
        optD = optim.Adam(netD.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.99))


        if cfg.TRAIN.NET_G != '':
              state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
              netG.load_state_dict(state_dict)
              print('Load G from: ', cfg.TRAIN.NET_G)

              istart = cfg.TRAIN.NET_G.rfind('_') + 1
              iend = cfg.TRAIN.NET_G.rfind('.')
              epoch = cfg.TRAIN.NET_G[istart:iend]
              epoch = int(epoch) + 1
              self.start_epoch = epoch

              state_dict = torch.load(cfg.TRAIN.NET_G.replace('G', 'D'), map_location=lambda storage, loc: storage)
              netD.load_state_dict(state_dict)
              print('Load D from: ', cfg.TRAIN.NET_G.replace('G', 'D'))


        netG = torch.nn.DataParallel(netG)
        netD = torch.nn.DataParallel(netD)
        image_encoder = torch.nn.DataParallel(image_encoder)

        return netG, netD, optG, optD, text_encoder, image_encoder


    def train(self):
        self.netG, self.netD, self.optG, self.optD, self.text_encoder, self.image_encoder = self.build_models()
        for epoch in range(self.start_epoch, self.max_epoch):
            # epoch = epoch + self.start_epoch
            for step, (shape, cap, cap_len, cls_id, key) in enumerate(self.data_loader):
              start_t = time.time()
              n_iter = (epoch * self.num_batches) + step
              shape = Variable(shape).to(self.device)


              sorted_cap_lens, sorted_cap_indices = torch.sort(cap_len, 0, True)

              #sort
              shapes = shape[sorted_cap_indices].squeeze().to(self.device)
              captions = cap[sorted_cap_indices].squeeze().to(self.device)
              class_ids = cls_id[sorted_cap_indices].squeeze().numpy()

              hidden = self.text_encoder.init_hidden(self.batch_size)
              # words_embs: batch_size x nef x seq_len
              # sent_emb: batch_size x nef
              words_embs, sent_emb = self.text_encoder(captions, sorted_cap_lens, hidden)

              d_loss = self._discriminator_train_step(shapes, sent_emb, words_embs, step, sorted_cap_lens, class_ids )
              self.exp.log_metric('d_loss', d_loss, step=step, epoch=epoch)


              if step % cfg.TRAIN.N_CRITIC == 0:
                g_loss= self._generator_train_step(sent_emb, words_embs, sorted_cap_lens, step, epoch)

                self.exp.log_metric('g_loss', g_loss, step=step, epoch=epoch)

              end_t = time.time()
              print('''[%d/%d][%d/%d] Time: %.2fs'''
              % (epoch, self.max_epoch, step, self.num_batches, end_t - start_t))

              if n_iter % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                fullpath = '%s/lean_%d_%d' % (self.shape_dir, epoch, step)
                noise = torch.randn(self.batch_size, self.nz).to(self.device)
                noise = noise.data.normal_(0.0, 1.0)
                shape, mu, logvar = self.netG(noise, sent_emb)
                # build_npy(shape.cpu().data, cap, self.ixtoword, fullpath)
                build_image(shape.cpu().data, cap, self.ixtoword, epoch, self.exp)

                torch.save(self.netG.module.state_dict(),'%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
                torch.save(self.netD.module.state_dict(),'%s/netD_epoch_%d.pth' % (self.model_dir, epoch))



    def _discriminator_train_step(self, real, sent_emb, words_emb, stage, sorted_cap_lens, class_ids):

        d_out_real = self.netD(real, sent_emb)
        loss_real = self.gan_loss(d_out_real, "dis_real")

        noise = torch.randn(self.batch_size, self.nz).to(self.device)
        noise = noise.data.normal_(0.0, 1.0)
        fake, mu, logvar= self.netG(noise, sent_emb)
        d_out_fake = self.netD(fake, sent_emb)
        loss_fake = self.gan_loss(d_out_fake, "dis_fake")

        d_worng_ans = self.netD(real[:(self.batch_size - 1)], sent_emb[1:self.batch_size])
        worng_fake = self.gan_loss(d_worng_ans, "dis_fake")

        loss = loss_real + loss_fake + worng_fake

        self.optD.zero_grad()
        loss.backward()
        self.optD.step()
        return loss.item()

    def _generator_train_step(self, sent_emb, words_emb, cap_lens, step, epoch):

        noise = torch.randn(self.batch_size, self.nz).to(self.device)
        fake_data, mu, logvar= self.netG(noise, sent_emb)
        fake_data = fake_data.to(self.device)

        if False:
          region_features, cnn_code = self.image_encoder(fake_data)

          match_labels = Variable(torch.LongTensor(range(self.batch_size)))
          s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, None, self.batch_size)
          s_loss = (s_loss0 + s_loss1) * 1
          w_loss0, w_loss1, _ = words_loss(region_features, words_emb, match_labels ,cap_lens, None, self.batch_size)
          w_loss = (w_loss0 + w_loss1) * 1


          g_out = self.netD(fake_data, sent_emb)
          kl_loss = KL_loss(mu, logvar)
          loss = self.gan_loss(g_out, "gen") + kl_loss + s_loss + w_loss

          self.exp.log_metric('kl_loss', kl_loss.item(), step=step, epoch=epoch)
          self.exp.log_metric('s_loss', s_loss.item(), step=step, epoch=epoch)

        else:
          g_out = self.netD(fake_data, sent_emb)
          kl_loss = KL_loss(mu, logvar)
          loss = self.gan_loss(g_out, "gen") + kl_loss


        self.optG.zero_grad()
        loss.backward()
        self.optG.step()
        return loss.item()

