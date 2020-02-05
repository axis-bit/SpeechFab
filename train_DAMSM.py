from comet_ml import Experiment
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from PIL import Image

from miscc.losses import sent_loss, words_loss
from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file
from miscc.utils import build_super_images

from datasets import Dataset
from networks.rnn_cnn import RNN_ENCODER, CNN_ENCODER
import pdb

import warnings
warnings.filterwarnings('ignore')

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

UPDATE_INTERVAL = 25


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/32_chair_table_local.yml', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()

    return args

def build_models():
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)


    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if torch.cuda.is_available():
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=torch.device('cpu'))
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name, map_location=torch.device('cpu'))
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        # start_epoch = int(start_epoch) + 1
        start_epoch = 1
        print('start_epoch', start_epoch)

    image_encoder = torch.nn.DataParallel(image_encoder)

    return text_encoder, image_encoder, labels, start_epoch

def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir, exp):
    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()


    for step, data in enumerate(dataloader):
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        shape, cap, cap_len, cls_id, key = data
        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_len, 0, True)

        #sort
        shapes = shape[sorted_cap_indices].squeeze()
        captions = cap[sorted_cap_indices].squeeze()
        cap_len = cap_len[sorted_cap_indices].squeeze()
        class_ids = cls_id[sorted_cap_indices].squeeze().numpy()


        if torch.cuda.is_available():
            shapes = shapes.cuda()
            captions = captions.cuda()
        #model
        words_features, sent_code = cnn_model(shapes)

        nef, att_sze = words_features.size(1), words_features.size(2)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, sorted_cap_lens, hidden)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 sorted_cap_lens, class_ids, batch_size)


        w_total_loss0 += w_loss0.item()
        w_total_loss1 += w_loss1.item()
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)

        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.item()
        s_total_loss1 += s_loss1.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()


        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0 / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1 / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0 / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1 / UPDATE_INTERVAL

            elapsed = time.time() - start_time

            exp.log_metric('s_cur_loss0', s_cur_loss0, step=step, epoch=epoch)
            exp.log_metric('s_cur_loss1', s_cur_loss1, step=step, epoch=epoch)

            exp.log_metric('w_cur_loss0', w_cur_loss0, step=step, epoch=epoch)
            exp.log_metric('w_cur_loss1', w_cur_loss1, step=step, epoch=epoch)


            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
        if step == 1:
            fullpath = '%s/attention_maps%d' % (image_dir, step)
            build_super_images(shapes.cpu().detach().numpy(), captions, cap_len, ixtoword, attn_maps, att_sze, exp, fullpath, epoch)

    return count


def evaluate(dataloader, cnn_model, rnn_model, batch_size, exp ,epoch):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        shape, cap, cap_len, cls_id, key = data

        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_len, 0, True)

        shapes = shape[sorted_cap_indices].squeeze()
        captions = cap[sorted_cap_indices].squeeze()
        class_ids = cls_id[sorted_cap_indices].squeeze().numpy()

        if torch.cuda.is_available():
            shapes = shapes.cuda()
            captions = captions.cuda()


        words_features, sent_code = cnn_model(shapes)
        nef, att_sze = words_features.size(1), words_features.size(2)


        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, sorted_cap_lens, hidden)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 sorted_cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).item()

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).item()

        if step == 10:
            break

    s_cur_loss = s_total_loss / step
    w_cur_loss = w_total_loss / step

    exp.log_metric('eva_s_cur_loss', s_cur_loss, epoch=epoch)
    exp.log_metric('eva_w_cur_loss', w_cur_loss, epoch=epoch)


    return s_cur_loss, w_cur_loss
class Subset(data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/outputs/%s_%s_%s' % \
        (cfg.OUTPUT, cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    cudnn.benchmark = True

    ##########################################################################

    batch_size = cfg.TRAIN.BATCH_SIZE
    data_path = cfg.DATA_DIR + cfg.DATASET_NAME
    dataset = Dataset(data_path, cfg.SPLIT, cfg.CATEGORY, cfg.CHANNEL)
    n_samples = len(dataset)
    train_size = int(n_samples * 0.80)

    subset1_indices = list(range(0,train_size))
    subset2_indices = list(range(train_size,n_samples))

    train_dataset = Subset(dataset, subset1_indices)
    val_dataset   = Subset(dataset, subset2_indices)

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    dataloader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # Train ##############################################################
    exp = Experiment(api_key="VeFWrxnaD0wAfQjDXoLy8gdiL", project_name=cfg.CONFIG_NAME, workspace="axis-bit")
    text_encoder, image_encoder, labels, start_epoch = build_models()
    para = list(text_encoder.parameters())

    print(len(dataloader_val))
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    try:
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            count = train(dataloader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir, exp)

            print('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98
            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.module.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')

            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, image_encoder, text_encoder, batch_size, exp, epoch)


                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')