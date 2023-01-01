'''
Author: huangqianfei
Date: 2023-01-01 14:16:58
LastEditTime: 2023-01-01 20:24:21
Description: 
'''
import os
import sys

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CUR_DIR + "/../")

import torch
import gensim
import subprocess
import collections

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformer import Constants


class Reader():
    def __init__(self, opt):
        self.opt = opt
        
        index = 0
        self._token_dict = {}
        f_r = open(opt.dict, "r")
        for line in f_r:
            token = line.strip()
            self._token_dict[token] = index
            index += 1
        f_r.close()

        self.reader = os.popen(f"cat {opt.train_data_path}")
        self.deque = collections.deque()
        self._lines = int(subprocess.getoutput(f"wc -l {opt.train_data_path}").strip().split(" ")[0])
        self.num = 0

    def __getitem__(self, index):
        if not self.deque:
            docs = self.reader.read(1024 * 1024 * 10).splitlines()
            docs = self.parse_data(docs)
            self.deque.extend(docs)

        item = self.deque.popleft()
        doc, label = item
        return doc, label

    def parse_data(self, docs):
        """get data index"""
        datas = []
        for line in docs:
            line = line.strip().split(" ")
            token_num_list = [self._token_dict[item] if item in self._token_dict else 1 for item in line]

            self.num += 1

            if self.num <= self.opt.train_pos:
                datas.append([token_num_list, 0])
            else:
                datas.append([token_num_list, 1])

        return datas

    def __len__(self):
        return self._lines

    
    def collate_fn(self, batch):
        """pad batch"""
        doc_list = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        max_len = max(len(inst) for inst in doc_list)

        batch_seq = np.array([inst + [Constants.PAD] * (max_len - len(inst))
            for inst in doc_list])

        batch_pos = np.array([
            [pos_i + 1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)] 
            for inst in batch_seq])

        batch_seq = torch.LongTensor(batch_seq)
        batch_pos = torch.LongTensor(batch_pos)
        batch_labels = torch.LongTensor(labels)

        return batch_seq, batch_pos, batch_labels

