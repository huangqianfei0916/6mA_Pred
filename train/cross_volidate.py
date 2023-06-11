import sys

sys.path.extend(["../../", "../", "./"])
import torch
import numpy as np
import pandas as pd
import time
import random
from sklearn import metrics
import argparse
import gensim
from torch import nn
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from net.lstm_attention import LSTM_attention
from data.dataset import dataset1
from sklearn.preprocessing import label_binarize


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


# 模型评估
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = torch.max(output, 1)

    num_correct = (pred_label == label).sum().item()

    return num_correct / total

# 训练模型
def train(model, train_data, valid_data, config, optimizer, criterion):
    model = model.to(config.device)
    prev_time = datetime.now()

    for epoch in range(config.num_epochs):
        model = model.train()
        train_loss = 0
        train_acc = 0

        for im, label in train_data:
            im = im.to(config.device)
            label = label.long().to(config.device)

            # forward
            output = model(im)

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
            predict_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)
            predict_pro = np.array([], dtype=float)
            flag=0
            for im, label in valid_data:
                with torch.no_grad():
                    im = im.to(config.device)
                    label = label.long().to(config.device)

                output = model(im)

                labels = label.data.cpu().numpy()
                predict = torch.max(output.data, 1)[1].cpu().numpy()
                pro = output.cpu().detach().numpy()
                if flag==0:
                    predict_pro=pro

                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predict)

                if flag==1:
                    predict_pro=np.vstack((predict_pro,pro))
                flag=1

                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
            acc = metrics.accuracy_score(labels_all, predict_all)

            y_one_hot=label_binarize(labels_all,np.arange(2))
            print(y_one_hot.shape)
            print(predict_pro.shape)
            print(predict_pro[:,1].shape)
            # exit()

            print("AUC:", metrics.roc_auc_score(y_one_hot, predict_pro[:,1], average='micro'))

            report = metrics.classification_report(labels_all, predict_all)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            print("ACC:",acc)
            print("MCC:{}".format(metrics.matthews_corrcoef(labels_all, predict_all)))

            print("report:",report)
            print("confusion:",confusion)
            print("--------------------------------")
            pd.DataFrame(labels_all).to_csv("../model/lab{}_{}.pkl".format(epoch,valid_acc / len(valid_data)),header=None)
            pd.DataFrame(predict_all).to_csv("../model/pre{}_{}.pkl".format(epoch,valid_acc / len(valid_data)),header=None)
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        torch.save(model.state_dict(), "../model/model{}_{}.pkl".format(epoch,valid_acc / len(valid_data)))



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

# 主函数
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parameter of train set
    parser.add_argument('-seed', default=2020, help="seed")
    parser.add_argument('-freeze', default=False)
    parser.add_argument('-embedding1',help="embedding model",required=True)
    parser.add_argument('-train_data_path', required=True)
    parser.add_argument('-train_pos', type=int, required=True)
    parser.add_argument('-train_neg', type=int, required=True)
    parser.add_argument('-batch_size', default=64)
    parser.add_argument('-fix_len', required=True,type=int)
    parser.add_argument('-learning_rate', default=0.001)
    parser.add_argument('-dropout', default=0.3)
    parser.add_argument('-num_classes', default=2)
    parser.add_argument('-hidden_dims', default=100)
    parser.add_argument('-rnn_layers', default=2)
    parser.add_argument('-num_epochs',type=int,default=15)
    parser.add_argument('-weight_decay', default=0.0)
    parser.add_argument('-init', default=True)
    parser.add_argument('-k', default=5)

    opt = parser.parse_args()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(opt)

    # gpu测试
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: ", torch.backends.cudnn.enabled)
    print('GPUs：', torch.cuda.device_count())

    torch.set_num_threads(4)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    w2id,vector=emb(opt.embedding1)
    word2vec = torch.Tensor(vector)

    # model = LSTM_attention(weight1=word2vec,opt=opt)

    d=dataset1(opt,w2id)

    # optimzier=torch.optim.SGD(model_parameters,lr=opt.learning_rate,weight_decay=opt.weight_decay)

    k=opt.k
    x, y = d.get_data(opt)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    # index = [i for i in range(len(x))]
    # random.shuffle(index)
    # x = x[index]
    # y = y[index]

    for i in range(k):

        X_train, y_train, X_valid, y_valid = d.get_k_fold_data(k, i, x, y)
        model = LSTM_attention(weight1=word2vec, opt=opt)
        if opt.freeze:
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        else:
            model_parameters = model.parameters()

        criterion = nn.CrossEntropyLoss()
        optimzier = torch.optim.Adam(model_parameters, lr=opt.learning_rate, weight_decay=opt.weight_decay)
        train_data = TensorDataset(X_train, y_train)
        valid_data = TensorDataset(X_valid, y_valid)

        train_loader = DataLoader(train_data, shuffle=True, batch_size=opt.batch_size)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=opt.batch_size)

        train(model, train_loader, valid_loader, opt, optimzier, criterion)
        model = LSTM_attention(weight1=word2vec,opt=opt)
