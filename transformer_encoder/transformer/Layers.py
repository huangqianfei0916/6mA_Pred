'''
Author: huangqianfei
Date: 2023-01-01 14:16:58
LastEditTime: 2023-01-01 15:34:12
Description: 
'''
import os
import sys

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CUR_DIR)

import torch.nn as nn
from SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):

        # 这里的enc_output是position_embedding,embedding,以及context vector的和向量
        # enc_slf_attn： 存储的是权重信息
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        # print(non_pad_mask)
        enc_output *= non_pad_mask
        # 这里将enc_output和经过前馈神经网络的输出相加
        enc_output = self.pos_ffn(enc_output)

        enc_output *= non_pad_mask


        return enc_output, enc_slf_attn

