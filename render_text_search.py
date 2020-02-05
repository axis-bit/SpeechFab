
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
from networks.rnn_cnn import RNN_ENCODER, CNN_ENCODER

from nltk.tokenize import RegexpTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Talk2Shape network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfg/32_chair_table_local.yml', type=str)
    parser.add_argument('--size', dest='size', default=64, type=int)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--text', type=str, help='manual seed')
    args = parser.parse_args()
    return args

def visualize_zs(zs, shapes, captions, keys, class_ids, ixtoword):
  fig, ax = plt.subplots(figsize=(50,50))
  colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]
  points = TSNE(n_components=2, random_state=0).fit_transform(zs)

  for i , (p, s, k, key, l) in enumerate(zip(points, shapes, captions, keys, class_ids)):
    sent = []
    for n in range(len(k)):
      s_id = k.numpy()[n]
      txt = ixtoword[int(s_id)]
      if(s_id != 0):
          sent.append(txt)

    l = 1

    img = plt.imread("./tools/shape_captcha/images/"+ str(key) +".png", format='png')
    imb0 = OffsetImage(img, zoom=0.1)
    imb0.image.axes = ax
    plt.tick_params(labelsize=40)
    plt.grid(which='major',color='black',linestyle='-')
    ab0  = AnnotationBbox(imb0, p, xybox=(40, -40), xycoords="data", boxcoords="offset points", pad=0.2, bboxprops=dict(edgecolor=colors[l]), arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab0)

    print(sent)
    plt.scatter(p[0], p[1], marker="${}$".format(' '.join(sent)), c=colors[l])
    ax.annotate(' '.join(sent), (p[0], p[1]))

  plt.savefig('./plts/search.png')



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


    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load("./networks/pretrain/text_encoder.pth", map_location=torch.device('cpu'))
    text_encoder.load_state_dict(state_dict)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(args.text.lower())

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

      hidden = text_encoder.init_hidden(1)
      words_embs, sent_emb = text_encoder(caps, cap_lens, hidden)
      print(sent_emb)

    for step, data in enumerate(dataloader):
        shape, cap, cap_len, cls_id, key = data

        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_len, 0, True)

        shapes = shape[sorted_cap_indices].squeeze()
        captions = cap[sorted_cap_indices].squeeze()
        cap_len = cap_len[sorted_cap_indices].squeeze()
        key = np.asarray(key)
        key = key[sorted_cap_indices].squeeze()
        class_ids = cls_id[sorted_cap_indices].squeeze().numpy()


        hidden = text_encoder.init_hidden(args.size)
        words_emb, sent_emb = text_encoder(captions, sorted_cap_lens, hidden)

        sent = sent_emb.cpu().detach().numpy()


        # np.savetxt("./plts/train.tsv", sent, delimiter="\t")
        # np.savetxt("./plts/label.tsv", key, delimiter="\t", fmt="%s")


        visualize_zs(sent, shapes, captions, key, class_ids, dataset.ixtoword)
        break
