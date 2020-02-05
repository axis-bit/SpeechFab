import os
import pprint
import pickle
import numpy as np
import pandas as pd
import torch.utils.data as data
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import time

from miscc.config import cfg, cfg_from_file


# id, modelId, description, category, topLevelSynsetId, subSynsetId

class Dataset(data.Dataset):
    def __init__(self, data_path, split, category_id, channel):
        self.data_path = data_path
        self.data = pd.read_csv(data_path + '/captions.csv')
        self.categories = self.data['category'].unique().tolist()
        self.shape_files = self.data['modelId'].unique().tolist()
        self.channel = channel
        self.category_id = category_id
        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(self.data_path, split)

        #=====================
        pprint.pprint("Captions items: " + str(len(self.captions)) )
        pprint.pprint("Category items: " + str(len(self.categories)) )
        pprint.pprint("Shape items: " + str(len(self.shape_files)) )
        pprint.pprint("Words: " + str(self.n_words))


    def load_captions(self):
        all_data = []
        for item in self.data.iterrows():

            dec = item[1][2]
            key = item[1][1]

            if self.category_id == 3:
                cat_id = self.categories.index(item[1][3])
            else:
                cat_id = self.shape_files.index(item[1][1])

            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(dec.lower())
            if len(tokens) < 25:
                all_data.append([tokens, key, cat_id])

        return all_data



    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions

        for sent in captions:
            for word in sent[0]:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t[0]:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t[0]:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_path, split):
        filepath = os.path.join( data_path + '/caption.pickle')
        if not os.path.isfile(filepath):
            data_list = self.load_captions()

            dl_len = len(data_list)
            train_len = round(dl_len * 0.90)

            train_list = data_list[:train_len]
            test_list = data_list[train_len:]
            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_list, test_list)

            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions, ixtoword, wordtoix, train_list, test_list], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                train_list, test_list= x[4], x[5]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)

        if split == "train":
            captions = train_captions
            filenames = train_list
        else:
            captions = test_captions
            filenames = test_list


        return filenames, captions, ixtoword, wordtoix, n_words

    def __getitem__(self, index):

        key = self.filenames[index][1]
        if self.channel == 1:
            shape = np.load(self.data_path + '/shapes/' + key + '.npy')[3]
        else:
            shape = np.load(self.data_path + '/shapes/' + key + '.npy')

        cls_id = self.filenames[index][2]

        # # sentence
        sent_caption = np.asarray(self.captions[index]).astype('int64')
        cap_len = len(sent_caption)
        cap = np.zeros((cfg.WORDS_NUM, 1), dtype='int64')
        cap[:cap_len, 0] = sent_caption

        return shape, cap, cap_len, cls_id, key

    def __len__(self):
        return len(self.captions)


class ShapeDataset(data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(data_path + '/captions.csv')
        self.shape_files = self.data['modelId'].unique().tolist()

    def __getitem__(self, index):
        shape = np.load(self.data_path + '/shapes/' + self.shape_files[index] + '.npy')
        return shape

    def __len__(self):
        return len(self.shape_files)