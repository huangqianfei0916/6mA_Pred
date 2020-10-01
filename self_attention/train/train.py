import sys

sys.path.extend(["../../", "../", "./"])
import torch
import time
import numpy as np
import gensim
from torch import nn
from datetime import datetime
from data.dataset1 import dataset1
from transformer.Models import Transformer
from transformer import Constants


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


# 模型评估

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = torch.max(output, 1)

    num_correct = (pred_label == label).sum().item()

    return num_correct / total



def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''

    insts=insts.tolist()
    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos
# ======================================================================================================================
# 训练模型
# ======================================================================================================================
def train(model, train_data, valid_data, config, optimizer, criterion):
    model = model.to(config.device)
    prev_time = datetime.now()

    for epoch in range(config.num_epochs):
        model = model.train()
        train_loss = 0
        train_acc = 0

        for im, label in train_data:

            a,b=collate_fn(im.detach().numpy())

            a = a.to(config.device)
            b = b.to(config.device)
            # im = im.to(config.device)
            label = label.long().to(config.device)

            # forward
            output = model(a,b)

            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            model = model.eval()
            for im, label in valid_data:
                with torch.no_grad():

                    a1, b1 = collate_fn(im.detach().numpy())

                    a1 = a1.to(config.device)
                    b1 = b1.to(config.device)
                    label = label.long().to(config.device)

                output = model(a1,b1)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)

def emb(model_path):

    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)

    wv = model.wv
    vocab_list = wv.index2word

    """将单词的下标和单词组成dict"""
    word2id = {}
    for id, word in enumerate(vocab_list):
        word2id[word] = id

    vectors = wv.vectors

    return word2id,vectors
# ======================================================================================================================
# 主函数
# ======================================================================================================================
import argparse
if __name__ == '__main__':
    # gpu测试
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: ", torch.backends.cudnn.enabled)
    print('GPUs：', torch.cuda.device_count())

    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    # parameter of train set
    parser.add_argument('-seed', default=2019, help="seed")
    parser.add_argument('-freeze', default=False)
    parser.add_argument('-embedding1', help="embedding model", required=True)
    parser.add_argument('-train_data_path', required=True)
    parser.add_argument('-train_pos', type=int, required=True)
    parser.add_argument('-train_neg', type=int, required=True)
    parser.add_argument('-batch_size', default=64)
    parser.add_argument('-test_data_path')
    parser.add_argument('-test_pos', type=int)
    parser.add_argument('-test_neg', type=int)
    parser.add_argument('-fix_len', required=True, type=int)
    parser.add_argument('-learning_rate', default=0.005)
    parser.add_argument('-dropout', default=0.3)
    parser.add_argument('-num_classes', default=2)
    parser.add_argument('-num_epochs', type=int, default=50)
    parser.add_argument('-weight_decay', default=0.0)
    parser.add_argument('-d_k', default=64)
    parser.add_argument('-d_v', default=64)
    parser.add_argument('-src_vocab_size', default=64)
    parser.add_argument('-d_model', default=100)
    parser.add_argument('-d_word_vec', default=100)
    parser.add_argument('-d_inner', default=512)
    parser.add_argument('-n_layers', default=1)
    parser.add_argument('-n_head', default=8)

    parser.add_argument('-init', default=True)

    config = parser.parse_args()
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(config)
    # config = DefaultConfig()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    w2id, vector = emb(config.embedding1)
    word2vec = torch.Tensor(vector)

    d=dataset1(config,w2id)

    if config.test_data_path:
        train_data = d.get_trainset(config)
        validate_data = d.get_testset(config)
    else:
        train_data, validate_data = d.get_splite_data(config)

    model = Transformer(
        config.src_vocab_size,
        config.fix_len,
        d_k=config.d_k,
        d_v=config.d_v,
        d_model=config.d_model,
        d_word_vec=config.d_word_vec,
        d_inner=config.d_inner,
        n_layers=config.n_layers,
        n_head=config.n_head,
        dropout=config.dropout)

    criterion = nn.CrossEntropyLoss()

    if config.freeze:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()

    optimzier = torch.optim.Adam(model_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)

    train(model, train_data, validate_data, config, optimzier, criterion)
