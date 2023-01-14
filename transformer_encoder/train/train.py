'''
Author: huangqianfei
Date: 2023-01-01 14:16:58
LastEditTime: 2023-01-14 15:45:52
Description: 
'''
import os
import sys

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CUR_DIR + "/../")

import torch
import time
import numpy as np
import gensim
import argparse
import torch.nn as nn

from torchsummary import summary
from torch.utils.data import DataLoader
from datetime import datetime
from data.dna_reader import Reader
from transformer.Models import Transformer
from transformer import Constants
from transformer.Optim import ScheduledOptim


def now():
    """get the current time"""
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def get_acc(output, label):
    """metrics"""
    total = output.shape[0]
    _, pred_label = torch.max(output, 1)
    num_correct = (pred_label == label).sum().item()

    return num_correct / total


def train(model, config, optimizer, criterion):
    """train"""
    model = model.to(config.device)
    prev_time = datetime.now()

    for epoch in range(config.num_epochs):
        model = model.train()
        train_loss = 0
        train_acc = 0
        train_data = Reader(config)
        train_loader = DataLoader(
            train_data, 
            shuffle = True, 
            batch_size = config.batch_size,
            collate_fn = train_data.collate_fn)


        val_data = Reader(config)
        val_loader = DataLoader(
            val_data, 
            shuffle = True, 
            batch_size = config.batch_size,
            collate_fn = train_data.collate_fn)

        for step, item in enumerate(train_loader):
            batch_seq, batch_pos, label = item

            batch_seq = batch_seq.to(config.device)
            batch_pos = batch_pos.to(config.device)
            label = label.long().to(config.device)

            # forward
            output = model(batch_seq, batch_pos)

            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        if val_loader is not None:
            valid_loss = 0
            valid_acc = 0
            model = model.eval()

            for step, item in enumerate(val_loader):
            
                with torch.no_grad():
                    batch_seq, batch_pos, label = item
                    batch_seq = batch_seq.to(config.device)
                    batch_pos = batch_pos.to(config.device)
                    label = label.long().to(config.device)

                output = model(batch_seq, batch_pos)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_loader),
                       train_acc / len(train_loader), valid_loss / len(val_loader),
                       valid_acc / len(val_loader)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_loader), train_acc / len(train_loader)))
        prev_time = cur_time
        print(epoch_str + time_str)


if __name__ == '__main__':
    """main"""
    # gpu test
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: ", torch.backends.cudnn.enabled)
    print('GPUs：', torch.cuda.device_count())

    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=2023, help="seed")
    parser.add_argument('-freeze', default=False)
    parser.add_argument('-dict', help="word dict", required=True)
    parser.add_argument('-train_data_path', required=True)
    parser.add_argument('-train_pos', type=int, required=True)
    parser.add_argument('-train_neg', type=int, required=True)
    parser.add_argument('-batch_size', default=64)
    parser.add_argument('-test_data_path')
    parser.add_argument('-test_pos', type=int)
    parser.add_argument('-test_neg', type=int)
    parser.add_argument('-fix_len', required=True, type=int)
    parser.add_argument('-learning_rate', default=0.005)
    parser.add_argument('-dropout', default=0.1)
    parser.add_argument('-num_classes', default=2)
    parser.add_argument('-num_epochs', type=int, default=50)
    parser.add_argument('-weight_decay', default=0.0)
    parser.add_argument('-d_k', default=64)
    parser.add_argument('-d_v', default=64)
    parser.add_argument('-src_vocab_size', default=10000)
    parser.add_argument('-d_model', default=512)
    parser.add_argument('-d_word_vec', default=512)
    parser.add_argument('-d_inner', default=512)
    parser.add_argument('-n_layers', default=1)
    parser.add_argument('-n_head', default=8)

    parser.add_argument('-init', default=True)

    config = parser.parse_args()
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(config)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    model = Transformer(
        n_src_vocab = config.src_vocab_size,
        len_max_seq = config.fix_len,
        d_k = config.d_k,
        d_v = config.d_v,
        d_model = config.d_model,
        d_word_vec = config.d_word_vec,
        d_inner = config.d_inner,
        n_layers = config.n_layers,
        n_head = config.n_head,
        dropout = config.dropout)

    criterion = nn.CrossEntropyLoss()

    if config.freeze:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()

    # summary(model, [(39,), (39,)], dtypes=[torch.long, torch.long], device='cpu')

    print("================================================")
    for key, value in model.named_parameters():
        print(key)
        print(value.shape)
    print("================================================")
    

    optimizer = ScheduledOptim(
        torch.optim.Adam(model_parameters, betas=(0.9, 0.98), eps = 1e-09), config.d_model, 100)

    train(model, config, optimizer, criterion)
