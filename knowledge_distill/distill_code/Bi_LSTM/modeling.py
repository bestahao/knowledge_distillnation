# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Bi-LSTM model & AutoEncoder model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchtext.vocab as vocab
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


class BiLSTM(nn.Module):
    """ Bi-LSTM + Max-Pooling + Tanh """
    def __init__(self, embedding_size, n_hidden, n_class, max_len, n_layers, cache_dir=f'/home/mist/from_bert/bert_embedding_layer.pth', embedding_vocab_size=30522, embedding_dim=768):
        super(BiLSTM, self).__init__()
        # use glove for embeddding
        # glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)
        # # add unknown vector
        # unknown_vector = torch.mean(glove.vectors, dim=0)
        # vectors = glove.vectors.detach()
        # vectors = torch.cat((unknown_vector.unsqueeze(0), vectors), 0)
        # self.embedding = nn.Embedding(vectors.size(0), vectors.size(1))
        # self.embedding.weight.data.copy_(vectors)
        
        # load bert embedding & freeze it
        layer = torch.nn.Embedding(embedding_vocab_size, embedding_dim)
        layer.load_state_dict(torch.load(cache_dir))
        self.embedding = layer
        for param in self.embedding.parameters():
            param.requires_grad = False 
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=n_hidden, bidirectional=True)
        self.max_pooling = nn.MaxPool1d(max_len)
        self.max_len = max_len
        self.fc = nn.Linear(n_hidden * 2, n_class)
        self.n_hidden = n_hidden
        self.n_layers = n_layers

    def forward(self, X, seq_lengths, device):
        # X: [batch_size, max_len]
        batch_size = X.shape[0]
        input_content = self.embedding(X) # # input : [batch_size, max_len, embedding_size]
        input_content = input_content.transpose(0, 1)  # input : [max_len, batch_size, embedding_size]
        input_content = pack_padded_sequence(input_content, seq_lengths, enforce_sorted=False)

        hidden_state = torch.randn(self.n_layers * 2, batch_size, self.n_hidden)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(self.n_layers * 2, batch_size, self.n_hidden)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        
        # set device
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
        input_content = input_content.to(device)
        
        hiddens, (_, _) = self.lstm(input_content, (hidden_state, cell_state))

        hiddens = pad_packed_sequence(hiddens) # [max_len, batch_size, n_hidden * 2]
        outputs = hiddens[0]
        
        # check size
        padding = outputs
        if outputs.shape[0] < self.max_len:
            padding = torch.zeros(self.max_len, batch_size, self.n_hidden*2)
            padding = padding.to(device)
            padding[:outputs.shape[0],:,:] = outputs
        
        outputs = padding.transpose(0, 2)
        outputs = self.max_pooling(outputs)  # [n_hidden * 2, batch_size, 1]
        outputs = outputs.transpose(0, 2)
        outputs = torch.tanh(outputs)  # [1, batch_size, n_hidden * 2]

        outputs_flat = outputs.transpose(0, 1)  # [1, batch_size, n_hidden * 2]
        outputs_flat = outputs_flat.flatten(start_dim=1, end_dim=2)  # [batch_size, 2 * n_hidden]
        # print('outputs_flat', outputs.shape)
        # print(outputs_flat.shape)
        logits = self.fc(outputs_flat)  # [batch_size, n_class]
        # print('logits', logits.shape)
        model = torch.softmax(logits, 1)
        # print('model', model, model.shape)

        return outputs, outputs_flat, logits, model


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers)

    def forward(self, X):
        # X : [max_len, batch_size, embedding_size]
        outputs, (hidden, cell) = self.rnn(X)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_size, hidden_size, n_layers):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers)
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, X, hidden, cell):
        # input = [batch size, embedding_size]
        X = X.unsqueeze(0) # [1, batch_size, embedding_size]
        output, (hidden, cell) = self.rnn(X, (hidden, cell)) # [1, batch_size, hidden_size]
        output = output.squeeze(0) # [batch_size, hidden_size]
        prediction = self.out(output) # [batch_size, output_dim]

        return prediction, hidden, cell


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, device, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size, embedding_size]
        # trg = [trg sent len, batch size, embedding_size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        embedding_size = trg.shape[2]
        batch_size = trg.shape[1]
        max_len = trg.shape[0]

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, embedding_size)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder.forward(src)
        hidden_out = hidden
        # first input to the decoder is the <sos> tokens
        input_ = torch.zeros(batch_size, embedding_size)
        input_ = input_.to(device)

        for t in range(max_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder.forward(input_, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = np.random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            input_ = trg[t, :] if teacher_force else output

        return hidden_out, outputs


class Student_FT(nn.Module):
    def __init__(self, bi_lstm, rnn_encoder):
        super(Student_FT, self).__init__()
        self.bi_lstm = bi_lstm
        self.rnn_encoder = rnn_encoder

    def forward(self, X, seq_lengths, device):
        student_hidden, student_flat_hidden, student_logits, student_repsd = self.bi_lstm(X, seq_lengths, device)
        hidden, _ = self.rnn_encoder(student_hidden)
        return hidden, student_logits, student_repsd


class Student_SPKD(nn.Module):
    def __init__(self, bi_lstm):
        super(Student_SPKD, self).__init__()
        self.bi_lstm = bi_lstm

    def forward(self, X, seq_lengths, device):
        hidden, flat_hidden, logits, model_pred = self.bi_lstm(X, seq_lengths, device)
        return hidden, flat_hidden, logits, model_pred


if __name__ == '__main__':
    test = 'no test'
    # test bi-lstm
    if test == 'bi-lstm':
        max_len = 100
        batch_size = 32
        embedding_size = 128
        n_class = 5
        n_hidden = 64
        bilstm = BiLSTM(embedding_size=embedding_size, n_hidden=n_hidden, n_class=n_class, max_len=max_len)
        input_data = np.random.uniform(0, 19, size=(batch_size, max_len, embedding_size))
        input_data = torch.from_numpy(input_data).float()
        print('input_data', input_data.shape)
        bilstm(input_data)
    elif test == 'autoencoder':
        max_len = 100
        batch_size = 32
        embedding_size = 128
        n_class = 5
        n_hidden = 64
        encoder = Encoder(embedding_size=embedding_size, hidden_size=n_hidden, n_layers=1)
        decoder = Decoder(output_dim=embedding_size, embedding_size=embedding_size, hidden_size=n_hidden, n_layers=1)
        autoencoder = AutoEncoder(encoder=encoder, decoder=decoder)
        X = torch.randn(max_len, batch_size, embedding_size)
        outputs = autoencoder(src=X, trg=X)
        mse_loss = nn.MSELoss()
        print(X.shape, outputs.shape)
        print(mse_loss(X, outputs))


