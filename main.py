from comet_ml import Experiment
import os
import sys
import random
import pprint
import time
import datetime
import dateutil.tz
import argparse
import torch
import numpy as np
import pandas as pd

from miscc.config import cfg, cfg_from_file
from datasets import Dataset
from trainer import condGANTrainer as trainer

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Talk2Shape network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/32_chair_table_local.yml', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()

    return args

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

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/outputs/%s_%s' % \
        (cfg.OUTPUT, cfg.CONFIG_NAME, timestamp)

    #==============================================================
    data_path = cfg.DATA_DIR + cfg.DATASET_NAME
    dataset = Dataset(data_path, cfg.SPLIT, cfg.CATEGORY, cfg.CHANNEL)

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, 1)
    algo.train()

