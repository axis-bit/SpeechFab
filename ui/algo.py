
from __future__ import print_function

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from miscc.config import cfg, cfg_from_file
from datasets import Dataset


from networks.rnn_cnn import RNN_ENCODER, CNN_ENCODER
from networks.dcgan import G_NET, D_NET

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudn
from skimage.transform import resize
from skimage.transform import resize
from skimage import measure
from scipy import ndimage


from nltk.tokenize import RegexpTokenizer
import json

class Eval(object):
  def __init__(self):
    #cfg_from_file("./cfgs/shape32.yml")
    cfg_from_file("./cfg/32_chair_table.yml")
    data_path = cfg.DATA_DIR + cfg.DATASET_NAME
    self.dataset = Dataset(data_path, cfg.SPLIT, cfg.CATEGORY, cfg.CHANNEL)
    self.wordtoix = self.dataset.wordtoix


    text_encoder = RNN_ENCODER(self.dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    #state_dict = torch.load("./data/models/text_encoder66.pth", map_location=lambda storage, loc: storage)
    state_dict = torch.load("./networks/pretrain/text_encoder.pth", map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder.eval()
    self.text_encoder = text_encoder


    self.netG = G_NET(4)
    # self.netG = torch.nn.DataParallel(netG)
    #state_dict = torch.load('./data/models/netG_epoch_39.pth', map_location=lambda storage, loc: storage)
    state_dict = torch.load("./networks/pretrain/netG_epoch.pth", map_location=lambda storage, loc: storage)
    self.netG.load_state_dict(state_dict)
    print('Load G from: ', cfg.TRAIN.NET_G)
    self.netG.eval()

    self.noise = torch.randn(1, 128)
    self.noise.data.normal_(0.5, 0.5)

  def create(self, text):

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())

    pprint.pprint(tokens)
    sent = []
    for item in tokens:
      try:
        w_id = self.wordtoix[item]
        sent.append(w_id)
      except:
        print("An exception occurred")

    if len(sent) >= 1:
      cap = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
      cap_len = len(sent)
      cap[:cap_len, 0] = sent
      cap = np.squeeze(cap)

      cap = np.array([cap])
      cap_lens = np.array([cap_len])

      caps = torch.from_numpy(cap)
      cap_lens = torch.from_numpy(cap_lens)

      hidden = self.text_encoder.init_hidden(1)
      words_embs, sent_emb = self.text_encoder(caps, cap_lens, hidden)
      fake_shapes, mu, var = self.netG(self.noise, sent_emb)

      return fake_shapes[0]



  def update(self, shape, size, one_flag, tri_flag):
    if shape is not None:

      a_x, a_y, a_z, b_y, b_x, b_z = [0,0,0,0,0,0]


      if one_flag:
        print("one_flag")
        blobs_labels = measure.label(np.round(shape[3]), connectivity=1, background=0)
        last = 0
        idx = 0
        for index, region in enumerate(measure.regionprops(blobs_labels)):
            if last < region.area:
                last = region.area
                idx = index
                a_x, a_y, a_z, b_y, b_x, b_z = region.bbox

                dd = np.where(blobs_labels == idx + 1, 1, 0)
                shape[3] = dd

      if tri_flag:
          print("tri_flag")
          dd = shape.transpose(1, 2, 3, 0)
          tri = dd[a_x:b_x, a_y:b_y, a_z:b_z, :]
          tri_tensor = tri.transpose(3, 0, 1, 2)



      res = (4, size, size, size)
      voxel_tensor = resize(shape, res)
      shape = voxel_tensor.transpose(1, 2, 3, 0)

      return shape

if __name__ == "__main__":
    algo = Eval()
    algo.create("this blue table .")