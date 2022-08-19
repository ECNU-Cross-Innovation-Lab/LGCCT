import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class CNN_BLSTM_SELF_ATTN(torch.nn.Module):
    def __init__(self, opt):
        super(CNN_BLSTM_SELF_ATTN, self).__init__()

        self.opt = opt
        lstm_input = 0
        model = []

        in_channel = self.opt.input_spec_size
        stride = 1
        for out_channel, kernel_size, padding in self.opt.cnn_layers:
            model += [nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding),
                      nn.ReLU()]
            lstm_input = in_channel = out_channel
        model += [Rearrange('b d l -> b l d')]

        self.model = nn.Sequential(*model)

        ###
        if self.opt.num_layers_lstm == 1:
            self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=self.opt.hidden_size_lstm,
                                num_layers=self.opt.num_layers_lstm, bidirectional=True, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=self.opt.hidden_size_lstm,
                                num_layers=self.opt.num_layers_lstm, bidirectional=True, dropout=0.5, batch_first=True)

        ## Transformer
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size_lstm*2,dim_feedforward=512,nhead=self.num_heads_self_attn)
        self.encoder_layer = self_attention_module(d_model=self.opt.hidden_size_lstm * 2,
                                                   dim_feedforward=self.opt.dim_feedfoward,
                                                   nhead=self.opt.num_heads_self_attn)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        # [batch, 74, 20]
        out = self.model(inputs)
        # [batch, 20, 64]
        out, _ = self.lstm(out)
        out = self.dropout(out)
        # out:[40, 20, 120]
        # out = self.encoder_layer(out)
        # stat:[40, 960]
        return out


class BLSTM_SELF_ATTN(torch.nn.Module):
    def __init__(self, opt, lstm_input):
        super(BLSTM_SELF_ATTN, self).__init__()

        self.opt = opt
        if self.opt.num_layers_lstm == 1:
            self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=self.opt.hidden_size_lstm,
                                num_layers=self.opt.num_layers_lstm, bidirectional=True, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=self.opt.hidden_size_lstm,
                                num_layers=self.opt.num_layers_lstm, bidirectional=True, dropout=0.5, batch_first=True)

        ## Transformer
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size_lstm*2,dim_feedforward=512,nhead=self.num_heads_self_attn)
        self.encoder_layer = self_attention_module(d_model=self.opt.hidden_size_lstm * 2,
                                                   dim_feedforward=self.opt.dim_feedfoward,
                                                   nhead=self.opt.num_heads_self_attn)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        out, _ = self.lstm(inputs)
        out = self.dropout(out)
        # out:[40, 33, 120]
        #out = self.encoder_layer(out)
        # stat:[40, 960]
        #out = torch.unsqueeze(out, 2)
        return out

class self_attention_module(torch.nn.Module):

    def __init__(self, d_model, dim_feedforward, nhead):
        super(self_attention_module, self).__init__()
        self.nhead = nhead

        model = [
            nn.Linear(d_model, dim_feedforward),
            nn.Tanh(),
            nn.Linear(dim_feedforward, nhead),
            nn.Softmax(dim=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        a = self.model(x)
        a = a.transpose(1, 2)
        h = a @ x
        z = torch.flatten(h, 1, -1)
        return h
