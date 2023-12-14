
def config_JointEmbeder():
    conf = {
        'gpu_id' : 1,
        # data_params
        'dataset_name':'CodeSearchDataset', # name of dataset to specify a data loader
        #training data
        'train_name':'train.name.h5',
        'train_token':'train.token.h5',
        'train_desc':'train.desc.h5',
        'train_api':'train.api.h5',
        # val data
        'val_name': 'val.name.h5',
        'val_token': 'val.token.h5',
        'val_desc': 'val.desc.h5',
        'val_api': 'val.api.h5',
        # test data
        'test_name':'test.name.h5',
        'test_token':'test.token.h5',
        'test_desc':'test.desc.h5',
        'test_api':'test.api.h5',

        #parameters
        'name_len': 6,
        'tok_len': 100,
        'desc_len': 35,
        'api_len': 30,

        'n_name_words': 7000, # len(vocabulary) + 1
        'n_token_words': 10000, # len(vocabulary) + 1
        'n_desc_words': 15000, # len(vocabulary) + 1
        'n_api_words': 20000, # len(vocabulary) + 1
        #vocabulary info
        'vocab_name':'vocab.name.json',
        'vocab_tokens':'vocab.token.json',
        'vocab_desc':'vocab.desc.json',
        'vocab_api':'vocab.api.json',

        #training_params
        'batch_size': 256,
        'nb_epoch': 300,
        #'optimizer': 'adam',
        'learning_rate':0.001, # try 1e-4(paper)
        'adam_epsilon':1e-8,
        'warmup_steps':5000,
        'fp16': False,
        'fp16_opt_level': 'O1', #For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
                        #"See details at https://nvidia.github.io/apex/amp.html"

    # model_params
        'use_desc_attn': 1,
        'use_tanh': 0,
        'emb_size': 512,
        'n_hidden': 512,#number of hidden dimension of code/desc representation
        'lstm_dims': 256,
        # recurrent
        'margin': 0.6,
        'sim_measure':'cos',#similarity measure: cos, poly, sigmoid, euc, gesd, aesd. see https://arxiv.org/pdf/1508.01585.pdf
                        #cos, poly and sigmoid are fast with simple dot, while euc, gesd and aesd are slow with vector normalization.
        'dropout':0.1,
    }
    return conf

