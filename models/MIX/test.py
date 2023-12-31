import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import traceback
import numpy as np
import argparse
import threading
import codecs
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch
import torch.nn.functional as F

import models, configs, data_loader
from modules import get_cosine_schedule_with_warmup
from utils import similarity, normalize
from data_loader import *


def test(config, model, device):
    logger.info('Test Begin...')

    model.eval()
    model.to(device)

    # load data
    data_path = args.data_path + args.dataset + '/'
    test_set = eval(config['dataset_name'])(config, data_path,
                                            config['test_graph'], config['n_node'],
                                            config['test_desc'], config['desc_len'])
    data_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=32,
                                              shuffle=False, drop_last=False, num_workers=1)
    # encode tokens and descs
    code_reprs, desc_reprs = [], []
    n_processed = 0
    for batch in data_loader:
        # batch[0:3]: init_input, adjmat, node_mask
        data_batch = [tensor.to(device) for tensor in batch]
        code_batch = [tensor.to(device) for tensor in batch[:3]]
        # batch[3:5]: good_desc, good_desc_len
        desc_batch = [tensor.to(device) for tensor in batch[3:5]]
        with torch.no_grad():
            code_repr = model.code_encoding(*code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.desc_encoding(*desc_batch).data.cpu().numpy().astype(np.float32)
            # code_repr = model.code_encoding(*code_batch).data.cpu()
            # desc_repr = model.desc_encoding(*desc_batch).data.cpu()

            loss = model(*data_batch)

            # [poolsize x hid_size]
            # aaa = F.cosine_similarity(code_repr, desc_repr)
            # normalize when sim_measure=='cos'
            code_repr = normalize(code_repr)
            desc_repr = normalize(desc_repr)
        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)  # +batch_size
        # if n_processed >= 100:
        #     break
    # code_reprs: [n_processed x n_hidden]
    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)

    # calculate similarity
    sum_1, sum_5, sum_10, sum_mrr = [], [], [], []
    test_sim_result, test_rank_result = [], []

    for i in tqdm(range(0, n_processed)):
        desc_vec = np.expand_dims(desc_reprs[i], axis=0)  # [1 x n_hidden]
        sims = np.dot(code_reprs, desc_vec.T)[:, 0]  # [n_processed]
        negsims = np.negative(sims)
        predict = np.argsort(negsims)

        # SuccessRate@k
        predict_1, predict_5, predict_10 = [int(predict[0])], [int(k) for k in predict[0:5]], [int(k) for k in
                                                                                               predict[0:10]]
        sum_1.append(1.0) if i in predict_1 else sum_1.append(0.0)
        sum_5.append(1.0) if i in predict_5 else sum_5.append(0.0)
        sum_10.append(1.0) if i in predict_10 else sum_10.append(0.0)
        # MRR
        predict_list = predict.tolist()
        rank = predict_list.index(i)
        sum_mrr.append(1 / float(rank + 1))

        # results need to be saved
        # predict_20 = [int(k) for k in predict[0:20]]
        # sim_20 = [sims[k] for k in predict_20]
        # test_sim_result.append(zip(predict_20, sim_20))
        # test_rank_result.append(rank + 1)

    logger.info(f'R@1={np.mean(sum_1)}, R@5={np.mean(sum_5)}, R@10={np.mean(sum_10)}, MRR={np.mean(sum_mrr)}')
    save_path = args.data_path + 'result/'
    sim_result_filename, rank_result_filename = 'sim.npy', 'rank.npy'
    # np.save(save_path + sim_result_filename, test_sim_result)
    # np.save(save_path + rank_result_filename, test_rank_result)


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='MIXEmbeder', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='dataset', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=300, help='epoch to reload from')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")

    parser.add_argument('--models_path', type=str, default='models_pretrain')
    # parser.add_argument('--models_path', type=str, default='models_fp16False_batch32_schedulerFalse')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = getattr(configs, 'config_' + args.model)()
    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu")
    ##### Define model ######
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config)  # initialize the model
    ckpt = f'./output/MIXEmbeder/pre_train/{args.models_path}/epo{args.reload_from}.h5'
    model.load_state_dict(torch.load(ckpt, map_location=device))

    test(config, model, device)
