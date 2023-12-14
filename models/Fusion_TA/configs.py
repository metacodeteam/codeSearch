
def config_MultiEmbeder():   
    conf = {

            'gpu_id': 0,
            'batch_size': 256,

            # added_params
            # 将输出经过Linear、Tanh、Linear层
            'transform_every_modal': 0,
            'transform_attn_out': 0,
            'use_tanh': 0,
            'save_graph_attn_weight': 0,
            'use_graph_attn': 1,
            'use_desc_attn': 1,
            'use_token_attn': 1,
            'use_ast_attn': 1,

            # # GGNN-cfg
            # 'state_dim': 512, # GGNN hidden state size
            # 'annotation_dim': 5,
            # 'n_edge_types': 1,
            # 'n_node': 30, # could be less than 512, like the maximum nodenum
            # 'n_steps': 5, # propogation steps number of GGNN
            # 'output_type': 'no_reduce',
            # 'n_layers': 1,
            # 'n_hidden': 512,
            # 'cfg_attn_mode': 'sigmoid_scalar',
            #
            # 'max_word_num': 65,
            # 'max_node_num': 30,

            # GGNN-cfg
            'state_dim': 512,  # GGNN hidden state size
            'annotation_dim': 350,
            'n_edge_types': 1,
            'n_node': 200,  # maximum nodenum
            'n_steps': 5,  # propogation steps number of GGNN
            'output_type': 'no_reduce',
            # 'output_type': 'sum',
            'n_layers': 1,
            'n_hidden': 512,
            'graph_attn_mode': 'sigmoid_scalar',
            # 'graph_attn_mode': 'softmax_scalar',
            'word_split': True,
            'pooling_type': 'max_pooling',  # ave_pooling
            'max_word_num': 30,
            'n_graph_words': 30000,

            # TreeLSTM
            'treelstm_cell_type': 'nary', # nary or childsum
            'n_ast_words': 50000,  #may be 20000?

            # Token and Description
            'desc_len': 35,
            'tok_len': 100,
            'n_desc_words': 15000,
            'n_token_words': 10000,

            # data_params
            'dataset_name':'CodeSearchDataset', # name of dataset to specify a data loader
            # training data
            'train_token':'train.token.h5',
            'train_ast':'train.ast.json',
            'train_cfg':'train.cfg.json',
            'train_desc':'train.desc.h5',
            # test data
            'test_token':'test.token.h5',
            'test_ast':'test.ast.json',
            'test_cfg':'test.cfg.json',
            'test_desc':'test.desc.h5',

            #val data
            'val_token': 'val.token.h5',
            'val_ast': 'val.ast.json',
            'val_cfg': 'val.cfg.json',
            'val_desc': 'val.desc.h5',
            # vocabulary info
            'vocab_token':'vocab.token.json',
            'vocab_ast':'vocab.ast.json',
            'vocab_desc':'vocab.desc.json',
                   
            # model_params
            'emb_size': 300,
            # recurrent  
            'margin': 0.6,
            'sim_measure':'cos',
            'dropout': 0.1,#try 0.3
            
                    
            # training_params            
            'nb_epoch': 300,
            #'optimizer': 'adamW',
            'learning_rate':0.001, # try 1e-4(paper)
            'adam_epsilon':1e-8,
            'warmup_steps':5000,
            'fp16': False,
            'fp16_opt_level': 'O1' #For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
                            #"See details at https://nvidia.github.io/apex/amp.html"

        
    }
    return conf

