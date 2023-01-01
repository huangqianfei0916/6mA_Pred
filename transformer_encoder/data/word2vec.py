'''
Author: huangqianfei
Date: 2023-01-01 14:16:58
LastEditTime: 2023-01-01 15:09:17
Description: 
'''
import argparse

from gensim.models import word2vec


def tomodel(train_word, iter1, sg, hs, window, size):
    sentences = word2vec.LineSentence(train_word)
    model = word2vec.Word2Vec(sentences, epochs=iter1, sg=sg, hs=hs, min_count=1, window=window, vector_size=size)
    model.wv.save_word2vec_format("word2vec.model", binary=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-word', required=True,help="word file")
    parser.add_argument('-iter', default=3)
    parser.add_argument('-sg', default=0)
    parser.add_argument('-hs', default=1)
    parser.add_argument('-window', default=3)
    parser.add_argument('-size', default=100)
    opt = parser.parse_args()
    print(opt)
    tomodel(opt.word, opt.iter, opt.sg, opt.hs, opt.window, opt.size)
    print("end..................")

