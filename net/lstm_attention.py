import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_attention(nn.Module):

    def __init__(self, opt, weight1):
        super(LSTM_attention, self).__init__()

        self.num_classes = opt.num_classes
        self.learning_rate = opt.learning_rate
        self.dropout = opt.dropout
        self.opt = opt
        self.weight1 = weight1

        self.w2v_vocab_size, self.w2v_dim = weight1.shape

        self.hidden_dims = opt.hidden_dims
        self.rnn_layers = opt.rnn_layers

        self.build_model()

    def build_model(self):
        if self.opt.init:
            self.w2v_embeddings = nn.Embedding(self.w2v_vocab_size, self.w2v_dim)
        else:
            self.w2v_embeddings = nn.Embedding(self.w2v_vocab_size, self.w2v_dim).from_pretrained(
                embeddings=self.weight1, freeze=self.opt.freeze)

        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )

        self.lstm_net = nn.LSTM(self.w2v_dim, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=True, batch_first=True)
        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes)
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden):

        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        out = lstm_tmp_out[0] + lstm_tmp_out[1]
        # 这里的out是将双向的信息融合

        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        lstm_hidden = lstm_hidden.unsqueeze(1)

        atten_h = self.attention_layer(lstm_hidden)
        m = nn.Tanh()(out)

        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_h, m.transpose(1, 2))
        softmax_w = F.softmax(atten_context, dim=-1)
        context = torch.bmm(softmax_w, out)

        result = context.squeeze(1)
        return result

    def forward(self, x):
        x = x.long()

        sen_w2v_input = self.w2v_embeddings(x)
        sen_input = sen_w2v_input

        output, (h_n, c_n) = self.lstm_net(sen_input)
        # output : [batch_size, len_seq, n_hidden * 2]

        h_n = h_n.permute(1, 0, 2)
        # h_n = torch.mean(h_n, dim=0, keepdim=True)

        atten_out = self.attention_net_with_w(output, h_n)
        return self.fc_out(atten_out)
