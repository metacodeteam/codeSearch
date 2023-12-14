import sys
import torch
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle

from utils import PAD_ID, UNK_ID, indexes2sent
import configs

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """

    def __init__(self, config, data_dir, f_tokens=None, max_token_len=None, f_descs=None, max_desc_len=None):
        self.max_desc_len = max_desc_len
        self.max_token_len = max_token_len

        print("Loading Data...")

        table_desc = tables.open_file(data_dir + f_descs)
        self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
        self.idx_descs = table_desc.get_node('/indices')[:]

        table_token = tables.open_file(data_dir + f_tokens)
        self.tokens = table_token.get_node('/phrases')[:].astype(np.long)
        self.idx_tokens = table_token.get_node('/indices')[:]

        # assert len(self.graph_dict) == self.idx_descs.shape[0]
        self.data_len = self.idx_descs.shape[0]
        print("{} entries".format(self.data_len))

    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = np.append(seq, [PAD_ID] * (maxlen - len(seq)))
        seq = seq[:maxlen].astype(np.int64)
        return seq

    def __getitem__(self, offset):

        # token
        len, pos = self.idx_tokens[offset][0], self.idx_tokens[offset][1]
        token_len = min(int(len), self.max_token_len)
        token = self.tokens[pos: pos + token_len]
        token = self.pad_seq(token, self.max_token_len)

        len, pos = self.idx_descs[offset][0], self.idx_descs[offset][1]
        good_desc_len = min(int(len), self.max_desc_len)
        good_desc = self.descs[pos: pos + good_desc_len]
        good_desc = self.pad_seq(good_desc, self.max_desc_len)

        rand_offset = random.randint(0, self.data_len - 1)
        while (rand_offset == offset):
            rand_offset = random.randint(0, self.data_len - 1)
        len, pos = self.idx_descs[rand_offset][0], self.idx_descs[rand_offset][1]
        bad_desc_len = min(int(len), self.max_desc_len)
        bad_desc = self.descs[pos: pos + bad_desc_len]
        bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

        return token, token_len, good_desc, good_desc_len, bad_desc, bad_desc_len

    def __len__(self):
        return self.data_len
