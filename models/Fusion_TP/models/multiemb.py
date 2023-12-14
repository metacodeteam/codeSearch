import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)
from modules import SeqEncoder, GGNN, TreeLSTM


class MultiEmbeder(nn.Module):
    def __init__(self, config):
        super(MultiEmbeder, self).__init__()
        self.conf = config

        self.margin = config['margin']
        self.emb_size = config['emb_size']
        self.n_hidden = config['n_hidden']
        self.dropout = config['dropout']

        self.n_desc_words = config['n_desc_words']
        self.n_token_words = config['n_token_words']
        self.graph_attn_mode = config['graph_attn_mode']

        self.graph_encoder = GGNN(self.conf)
        self.token_encoder = SeqEncoder(self.n_token_words, self.emb_size, self.n_hidden)
        self.desc_encoder = SeqEncoder(self.n_desc_words, self.emb_size, self.n_hidden)

        self.graph_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.graph_attn_scalar = nn.Linear(self.n_hidden, 1)
        self.token_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.token_attn_scalar = nn.Linear(self.n_hidden, 1)
        self.desc_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.desc_attn_scalar = nn.Linear(self.n_hidden, 1)
        self.attn_modal_fusion = nn.Linear(self.n_hidden * 2, self.n_hidden)

        self.linear_attn_out = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                             nn.Tanh(),
                                             nn.Linear(self.n_hidden, self.n_hidden))

        if self.conf['transform_every_modal']:
            self.linear_single_modal = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                                     nn.Tanh(),
                                                     nn.Linear(self.n_hidden, self.n_hidden))
        if self.conf['transform_attn_out']:
            self.linear_attn_out = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                                 nn.Tanh(),
                                                 nn.Linear(self.n_hidden, self.n_hidden))

        if self.conf['save_graph_attn_weight']:
            self.graph_attn_weight_torch = []
            self.graph_node_mask_torch = []

    def code_encoding(self, token, token_len, graph_init_input_batch, graph_adjmat_batch,
                      graph_node_mask):

        batch_size = graph_node_mask.size()[0]

        ''' Token Embedding w.Attention '''
        token_enc_hidden = self.token_encoder.init_hidden(batch_size)
        token_feat, token_enc_hidden = self.token_encoder(token, token_len, token_enc_hidden)
        token_enc_hidden = token_enc_hidden[0]

        if self.conf['transform_every_modal']:
            token_enc_hidden = torch.tanh(
                self.linear_single_modal(
                    F.dropout(token_enc_hidden, self.dropout, training=self.training)
                )
            )
        elif self.conf['use_tanh']:
            token_enc_hidden = torch.tanh(token_enc_hidden)

        if self.conf['use_token_attn']:
            seq_len = token_feat.size()[1]

            device = torch.device(f"cuda:{self.conf['gpu_id']}" if torch.cuda.is_available() else "cpu")
            unpack_len_list = token_len.long().to(device)
            range_tensor = torch.arange(seq_len).to(device)
            mask_1forgt0 = range_tensor[None, :] < unpack_len_list[:, None]
            mask_1forgt0 = mask_1forgt0.reshape(-1, seq_len)

            token_sa_tanh = torch.tanh(
                self.token_attn(token_feat.reshape(-1, self.n_hidden)))  # [(batch_sz * seq_len) x n_hidden]
            token_sa_tanh = F.dropout(token_sa_tanh, self.dropout, training=self.training)
            token_sa_tanh = self.token_attn_scalar(token_sa_tanh).reshape(-1, seq_len)  # [batch_sz x seq_len]
            token_feat = token_feat.reshape(-1, seq_len, self.n_hidden)

            tok_feat_attn = None
            for _i in range(batch_size):
                token_sa_tanh_one = torch.masked_select(token_sa_tanh[_i, :], mask_1forgt0[_i, :]).reshape(1, -1)
                # attn_w_one: [1 x 1 x seq_len]
                attn_w_one = F.softmax(token_sa_tanh_one, dim=1).reshape(1, 1, -1)

                # attn_feat_one: [1 x seq_len x n_hidden]
                attn_feat_one = torch.masked_select(token_feat[_i, :, :].reshape(1, seq_len, self.n_hidden),
                                                    mask_1forgt0[_i, :].reshape(1, seq_len, 1)).reshape(1, -1,
                                                                                                        self.n_hidden)
                # out_to_cat: [1 x n_hidden]
                out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.n_hidden)
                # self_attn_cfg_feat: [batch_sz x n_hidden]
                tok_feat_attn = out_to_cat if tok_feat_attn is None else torch.cat(
                    (tok_feat_attn, out_to_cat), 0)
        else:
            tok_feat_attn = token_enc_hidden.reshape(batch_size, self.n_hidden)

        if self.conf['transform_attn_out']:
            tok_feat_attn = torch.tanh(
                self.linear_attn_out(
                    F.dropout(tok_feat_attn, self.dropout, training=self.training)
                )
            )
        elif self.conf['use_tanh']:
            tok_feat_attn = torch.tanh(tok_feat_attn)




        ''' PDG Embedding w.Attention '''
        # graph_feat: [batch_size x n_node x state_dim]
        graph_feat = self.graph_encoder(graph_init_input_batch, graph_adjmat_batch,
                                        graph_node_mask)  # forward(annotation, A)

        node_num = graph_feat.size()[1]  # n_node

        graph_feat = graph_feat.reshape(-1, node_num, self.n_hidden)
        # mask_1forgt0: [batch_size x n_node]
        mask_1forgt0 = graph_node_mask.bool().reshape(-1, node_num)

        if self.conf['transform_every_modal']:
            graph_feat = torch.tanh(
                self.linear_single_modal(F.dropout(graph_feat, self.dropout, training=self.training)))
        elif self.conf['use_tanh']:
            graph_feat = torch.tanh(graph_feat)

        if self.conf['use_graph_attn']:
            graph_sa_tanh = torch.tanh(
                self.graph_attn(graph_feat.reshape(-1, self.n_hidden)))  # [(batch_size * n_node) x n_hidden]
            graph_sa_tanh = F.dropout(graph_sa_tanh, self.dropout, training=self.training)
            graph_sa_tanh = self.graph_attn_scalar(graph_sa_tanh).reshape(-1, node_num)

            graph_feat = graph_feat.reshape(-1, node_num, self.n_hidden)
            batch_size = graph_feat.size()[0]

            graph_feat_attn = None
            for _i in range(batch_size):
                graph_sa_tanh_one = torch.masked_select(graph_sa_tanh[_i, :], mask_1forgt0[_i, :]).reshape(1, -1)

                if self.graph_attn_mode == 'sigmoid_scalar':
                    # attn_w_one: [1 x 1 x real_node_num]
                    attn_w_one = torch.sigmoid(graph_sa_tanh_one).reshape(1, 1, -1)
                else:
                    attn_w_one = F.softmax(graph_sa_tanh_one, dim=1).reshape(1, 1, -1)

                if self.conf['save_graph_attn_weight']:
                    self.graph_attn_weight_torch.append(attn_w_one.detach().reshape(1, -1).cpu())
                    self.graph_node_mask_torch.append(mask_1forgt0[_i, :].detach().reshape(1, -1).cpu())

                # attn_feat_one: [1 x real_node_num x n_hidden]
                attn_feat_one = torch.masked_select(graph_feat[_i, :, :].reshape(1, node_num, self.n_hidden),
                                                    mask_1forgt0[_i, :].reshape(1, node_num, 1)).reshape(1, -1,
                                                                                                         self.n_hidden)
                out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.n_hidden)
                graph_feat_attn = out_to_cat if graph_feat_attn is None else torch.cat(
                    (graph_feat_attn, out_to_cat), 0)
        else:
            graph_feat_attn = graph_feat.reshape(batch_size, self.n_hidden)
        if self.conf['transform_attn_out']:
            graph_feat_attn = torch.tanh(
                self.linear_attn_out(
                    F.dropout(graph_feat_attn, self.dropout, training=self.training))
            )
        elif self.conf['use_tanh']:
            graph_feat_attn = torch.tanh(graph_feat_attn)

        # concat_feat: [batch_size x (n_hidden * 3)]
        concat_feat = torch.cat((tok_feat_attn, graph_feat_attn), 1)
        # code_feat: [batch_size x n_hidden]
        code_feat = torch.tanh(
            self.attn_modal_fusion(concat_feat)
        ).reshape(-1, self.n_hidden)

        return code_feat

    def desc_encoding(self, desc, desc_len):
        batch_size = desc.size()[0]
        desc_enc_hidden = self.desc_encoder.init_hidden(batch_size)
        # desc_enc_hidden: [2 x batch_size x n_hidden]
        desc_feat, desc_enc_hidden = self.desc_encoder(desc, desc_len, desc_enc_hidden)
        # desc_feat: [batch_size x n_hidden]
        desc_enc_hidden = desc_enc_hidden[0]

        if self.conf['transform_every_modal']:
            desc_enc_hidden = torch.tanh(
                self.linear_single_modal(
                    F.dropout(desc_enc_hidden, self.dropout, training=self.training)
                )
            )
        elif self.conf['use_tanh']:
            desc_enc_hidden = torch.tanh(desc_enc_hidden)

        if self.conf['use_desc_attn']:
            seq_len = desc_feat.size()[1]

            device = torch.device(f"cuda:{self.conf['gpu_id']}" if torch.cuda.is_available() else "cpu")
            unpack_len_list = desc_len.long().to(device)
            range_tensor = torch.arange(seq_len).to(device)
            mask_1forgt0 = range_tensor[None, :] < unpack_len_list[:, None]
            mask_1forgt0 = mask_1forgt0.reshape(-1, seq_len)

            desc_sa_tanh = torch.tanh(
                self.desc_attn(desc_feat.reshape(-1, self.n_hidden)))  # [(batch_sz * seq_len) x n_hidden]
            desc_sa_tanh = F.dropout(desc_sa_tanh, self.dropout, training=self.training)
            desc_sa_tanh = self.desc_attn_scalar(desc_sa_tanh).reshape(-1, seq_len)  # [batch_sz x seq_len]
            desc_feat = desc_feat.reshape(-1, seq_len, self.n_hidden)

            self_attn_desc_feat = None
            for _i in range(batch_size):
                desc_sa_tanh_one = torch.masked_select(desc_sa_tanh[_i, :], mask_1forgt0[_i, :]).reshape(1, -1)
                # attn_w_one: [1 x 1 x seq_len]
                attn_w_one = F.softmax(desc_sa_tanh_one, dim=1).reshape(1, 1, -1)

                # attn_feat_one: [1 x seq_len x n_hidden]
                attn_feat_one = torch.masked_select(desc_feat[_i, :, :].reshape(1, seq_len, self.n_hidden),
                                                    mask_1forgt0[_i, :].reshape(1, seq_len, 1)).reshape(1, -1,
                                                                                                        self.n_hidden)
                # out_to_cat: [1 x n_hidden]
                out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.n_hidden)
                # self_attn_cfg_feat: [batch_sz x n_hidden]
                self_attn_desc_feat = out_to_cat if self_attn_desc_feat is None else torch.cat(
                    (self_attn_desc_feat, out_to_cat), 0)
        else:
            self_attn_desc_feat = desc_enc_hidden.reshape(batch_size, self.n_hidden)

        if self.conf['transform_attn_out']:
            self_attn_desc_feat = torch.tanh(
                self.linear_attn_out(
                    F.dropout(self_attn_desc_feat, self.dropout, training=self.training)
                )
            )
        elif self.conf['use_tanh']:
            self_attn_desc_feat = torch.tanh(self_attn_desc_feat)

        self_attn_desc_feat = torch.tanh(self_attn_desc_feat)

        # desc_feat: [batch_ size x n_hidden]
        return self_attn_desc_feat

    def forward(self, tokens, tok_len, cfg_init_input, cfg_adjmat, cfg_node_mask, desc_anchor,
                desc_anchor_len, desc_neg, desc_neg_len):
        # code_repr: [batch_size x n_hidden]
        code_repr = self.code_encoding(tokens, tok_len, cfg_init_input, cfg_adjmat, cfg_node_mask)
        # desc_repr: [batch_size x n_hidden]
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        # sim: [batch_size]
        anchor_sim = F.cosine_similarity(code_repr, desc_anchor_repr)
        neg_sim = F.cosine_similarity(code_repr, desc_neg_repr)

        loss = (self.margin - anchor_sim + neg_sim).clamp(min=1e-6).mean()

        return loss