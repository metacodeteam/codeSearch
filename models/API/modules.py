import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

class BOWEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(BOWEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        #self.word_weights = get_word_weights(vocab_size) 
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        
    def forward(self, input, input_len=None): 
        batch_size, seq_len =input.size()
        embedded = self.embedding(input)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        embedded= F.dropout(embedded, 0.25, self.training) # [batch_size x seq_len x emb_size]
        
        # try to use a weighting scheme to summarize bag of word embeddings: 
        # for example, a smooth inverse frequency weighting algorithm: https://github.com/peter3125/sentence2vec/blob/master/sentence2vec.py
        # word_weights = self.word_weights(input) # [batch_size x seq_len x 1]
        # embeded = word_weights*embedded 
        
        # max pooling word vectors
        output_pool = F.max_pool1d(embedded.transpose(1,2), seq_len).squeeze(2) # [batch_size x emb_size]
        encoding = output_pool #torch.tanh(output_pool)        
        return encoding


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        for name, param in self.lstm.named_parameters(): # initialize the gate weights
            if 'weight' in name or 'bias' in name: 
                param.data.uniform_(-0.1, 0.1)

    def forward(self, inputs, input_lens=None): 
        batch_size, seq_len=inputs.size()
        inputs = self.embedding(inputs)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        inputs = F.dropout(inputs, 0.25, self.training)

        if input_lens is not None:# sort and pack sequence
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)

        hids, (h_n, c_n) = self.lstm(inputs) # hids:[b x seq x hid_sz*2](biRNN) 
        
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)   
            hids = F.dropout(hids, p=0.25, training=self.training)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, 2, batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
############commenting the following line significantly improves the performance, why? #####################################
        #h_n = h_n.transpose(1, 0).contiguous() [batch_size x n_dirs x hid_sz]
        encoding = h_n.view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]
        #pooled_encoding = F.max_pool1d(hids.transpose(1,2), seq_len).squeeze(2) # [batch_size x hid_size*2]
        #encoding = torch.tanh(pooled_encoding)

        return encoding #pooled_encoding

class SeqEncoder2(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder2, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)

        self.init_xavier_linear(self.embedding, init_bias=False)

        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0, batch_first=True, bidirectional=False)

    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().requires_grad_(), # rnn_type == 'LSTM'
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().requires_grad_())


    def forward(self, inputs, input_lens=None, hidden=None): 
        batch_size, seq_len = inputs.size()
        inputs = self.embedding(inputs)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        #inputs = F.dropout(inputs, 0.1, self.training) # mark.
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
            
        hids, (h_n, c_n) = self.lstm(inputs, hidden) # hids:[b x seq x hid_sz*2](biRNN) 
        
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)   
            #hids = F.dropout(hids, p=0.1, training=self.training) # mark.
            hids = hids.index_select(0, inv_indices) # [batch_sz x seq_len x hid_sz]
            h_n = h_n.index_select(1, inv_indices)
            c_n = c_n.index_select(1, inv_indices)

        h_n = h_n[0] # [batch_sz x hid_sz] n_layers==1 and n_dirs==1
        c_n = c_n[0] 

        return hids, (h_n, c_n)

    
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)    
    

def get_word_weights(vocab_size, padding_idx=0):
    '''contruct a word weighting table '''
    def cal_weight(word_idx):
        return 1-math.exp(-word_idx)
    weight_table = np.array([cal_weight(w) for w in range(vocab_size)])
    if padding_idx is not None:        
        weight_table[padding_idx] = 0. # zero vector for padding dimension
    return torch.FloatTensor(weight_table)
