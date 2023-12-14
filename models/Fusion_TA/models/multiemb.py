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

        self.ast_encoder = TreeLSTM(self.conf)
        self.token_encoder = SeqEncoder(self.n_token_words, self.emb_size, self.n_hidden)
        self.desc_encoder = SeqEncoder(self.n_desc_words, self.emb_size, self.n_hidden)

        self.ast_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.ast_attn_scalar = nn.Linear(self.n_hidden, 1)
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


    def code_encoding(self, token, token_len, tree, tree_node_num):
        
        batch_size = token.size()[0]

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







        ''' AST Embedding w.Attention '''
        # tree: contain ['graph', 'mask', 'wordid', 'label']
        ast_enc_hidden = self.ast_encoder.init_hidden(tree.graph.number_of_nodes()) # use all_node_num to initialize h_0, c_0
        # all_node_h/c_in_batch: [all_node_num_in_batch x hidden_size]
        all_node_h_in_batch, all_node_c_in_batch = self.ast_encoder(tree, ast_enc_hidden)

        if self.conf['transform_every_modal']:
            all_node_h_in_batch = torch.tanh(
                self.linear_single_modal(F.dropout(all_node_h_in_batch, self.dropout, training=self.training)))
        elif self.conf['use_tanh']:
            all_node_h_in_batch = torch.tanh(all_node_h_in_batch)

        if self.conf['use_ast_attn']:
            ast_feat_attn = None
            add_up_node_num = 0
            for _i in range(batch_size):
                # this_sample_h: [this_sample_node_num x hidden_size]
                this_sample_h = all_node_h_in_batch[add_up_node_num: add_up_node_num + tree_node_num[_i]]
                add_up_node_num += tree_node_num[_i]

                node_num = tree_node_num[_i] # this_sample_node_num
                ast_sa_tanh = torch.tanh(self.ast_attn(this_sample_h.reshape(-1, self.n_hidden)))
                ast_sa_tanh = F.dropout(ast_sa_tanh, self.dropout, training=self.training)
                ast_sa_before_softmax = self.ast_attn_scalar(ast_sa_tanh).reshape(1, node_num)
                # ast_attn_weight: [1 x this_sample_node_num]
                ast_attn_weight = F.softmax(ast_sa_before_softmax, dim=1)
                # ast_attn_this_sample_h: [1 x n_hidden]
                ast_attn_this_sample_h = torch.bmm(ast_attn_weight.reshape(1, 1, node_num),
                                                    this_sample_h.reshape(1, node_num, self.n_hidden)).reshape(1, self.n_hidden)
                # ast_feat_attn: [batch_size x n_hidden]
                ast_feat_attn = ast_attn_this_sample_h if ast_feat_attn is None else torch.cat(
                    (ast_feat_attn, ast_attn_this_sample_h), 0)
        else:
            ast_feat_attn = all_node_h_in_batch.reshape(batch_size, self.n_hidden)

        if self.conf['transform_attn_out']:
            ast_feat_attn = torch.tanh(
                self.linear_attn_out(
                    F.dropout(ast_feat_attn, self.dropout, training=self.training))
            )
        elif self.conf['use_tanh']:
            ast_feat_attn = torch.tanh(ast_feat_attn)


        # concat_feat: [batch_size x (n_hidden * 3)]
        concat_feat = torch.cat((tok_feat_attn, ast_feat_attn), 1)
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

        # self_attn_desc_feat = torch.tanh(self_attn_desc_feat)

        self_attn_desc_feat = torch.tanh(self_attn_desc_feat)

        # desc_feat: [batch_ size x n_hidden]
        return self_attn_desc_feat
    
    
    def forward(self, tokens, tok_len, tree, tree_node_num, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        # code_repr: [batch_size x n_hidden]  
        code_repr = self.code_encoding(tokens, tok_len, tree, tree_node_num)
        # desc_repr: [batch_size x n_hidden]  
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        # sim: [batch_size]
        anchor_sim = F.cosine_similarity(code_repr, desc_anchor_repr)
        neg_sim = F.cosine_similarity(code_repr, desc_neg_repr) 
        
        loss = (self.margin-anchor_sim+neg_sim).clamp(min=1e-6).mean()
        
        return loss