import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

from tqdm import *
import random
import time
from datetime import datetime
import argparse
random.seed(42)
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import logging
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(message)s")

import models
from modules import get_cosine_schedule_with_warmup
from data_loader import *

def normalize(data):
    """normalize matrix by rows"""
    return data/np.linalg.norm(data,axis=1,keepdims=True)


def train(args):
    fh = logging.FileHandler(f"./output/{args.model}/{args.dataset}/logs.txt")
    # create file handler which logs even debug messages
    logger.addHandler(fh)  # add the handlers to the logger
    tb_writer = SummaryWriter("logs") if args.visual else None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    def save_model(model, epoch):
        torch.save(model.state_dict(), f'./output/{args.model}/{args.dataset}/models/epo{epoch}.h5')

    def load_model(model, epoch, to_device):
        assert os.path.exists(
            f'./output/{args.model}/{args.dataset}/models/epo{epoch}.h5'), f'Weights at epoch {epoch} not found'
        model.load_state_dict(
            torch.load(f'./output/{args.model}/{args.dataset}/models/epo{epoch}.h5', map_location=to_device))

    config = getattr(configs, 'config_' + args.model)()
    print(config)

    # load data
    data_path = args.data_path + args.dataset + '/'
    train_set = eval(config['dataset_name'])(config, data_path,
                                             config['train_graph'], config['n_node'],
                                             config['train_desc'], config['desc_len'])

    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'],
                                              shuffle=True, drop_last=False, num_workers=0)

    # define the models
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config)  # initialize the model
    if args.reload_from > 0:
        load_model(model, args.reload_from, device)
    logger.info('done')
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config['warmup_steps'],
        num_training_steps=len(data_loader) * config[
            'nb_epoch'])  # do not forget to modify the number when dataset is changed

    print('---model parameters---')
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)

    n_iters = len(data_loader)
    itr_global = args.reload_from + 1
    for epoch in range(int(args.reload_from) + 1, config['nb_epoch'] + 1):
        itr_start_time = time.time()
        losses = []
        train_losses = []
        for batch in data_loader:

            model.train()
            batch_gpu = [tensor.to(device) for tensor in batch]
            loss = model(*batch_gpu)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            losses.append(loss.item())
            train_losses.append(loss.item())
            if itr_global % args.log_every == 0:
                elapsed = time.time() - itr_start_time
                logger.info('epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f' %
                            (epoch, config['nb_epoch'], itr_global % n_iters, n_iters, elapsed, np.mean(losses)))

                losses = []
                itr_start_time = time.time()
            itr_global = itr_global + 1
            time.sleep(0.003)
        logger.info('epo:[%d/%d] train_Loss=%.5f' %
                    (epoch, config['nb_epoch'],  np.mean(train_losses)))

        # save every epoch
        # if epoch >= 90:
        #     if epoch % 5 == 0:
        #         save_model(model, epoch)
        if epoch % 50 == 0:
            save_model(model, epoch)

        if epoch % args.val_every == 0:
            logger.info("validating..")
            model.eval()
            val_set = eval(config['dataset_name'])(config, data_path,
                                                    config['val_graph'], config['n_node'],
                                                    config['val_desc'], config['desc_len'])
            data_loader_val = torch.utils.data.DataLoader(dataset=val_set, batch_size=32,
                                                      shuffle=False, drop_last=False,
                                                      num_workers=0)

            code_reprs, desc_reprs = [], []
            n_processed = 0
            val_losses = []
            for batch_val in data_loader_val:
                # batch[0:7]: tokens, tok_len, tree, tree_node_num, init_input, adjmat, node_mask
                code_batch = [tensor.to(device) for tensor in batch_val[:3]]
                # batch[7:9]: good_desc, good_desc_len
                desc_batch = [tensor.to(device) for tensor in batch_val[3:5]]
                val_loss = model.forward(*[tensor.to(device) for tensor in batch_val[:]])

                val_losses.append(val_loss.item())

                with torch.no_grad():
                    code_repr = model.code_encoding(*code_batch).data.cpu().numpy().astype(np.float32)
                    desc_repr = model.desc_encoding(*desc_batch).data.cpu().numpy().astype(
                        np.float32)  # [poolsize x hid_size]
                    # [poolsize x hid_size]
                    code_repr = normalize(code_repr)
                    desc_repr = normalize(desc_repr)
                code_reprs.append(code_repr)
                desc_reprs.append(desc_repr)
                n_processed += batch_val[0].size(0)  # +batch_size
                # if n_processed >= 2000:
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
            logger.info(f'epo:{epoch}, R@1={np.mean(sum_1)}, R@5={np.mean(sum_5)}, R@10={np.mean(sum_10)}, MRR={np.mean(sum_mrr)}')
            logger.info('epo:[%d/%d] val_Loss=%.5f' %
                        (epoch, config['nb_epoch'],  np.mean(val_losses)))



            if tb_writer is not None:

                tb_writer.add_scalars(main_tag='loss',
                                      tag_scalar_dict={'train_loss': np.mean(train_losses),
                                                       'val_loss': np.mean(val_losses)},
                                      global_step=epoch)

                tb_writer.add_scalars(main_tag='result',
                                      tag_scalar_dict={'R@1': np.mean(sum_1),
                                                       'R@5': np.mean(sum_5),
                                                       'R@10': np.mean(sum_10),
                                                       'MRR': np.mean(sum_mrr)},
                                      global_step=epoch)

        model.train()



    
def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='/data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='PDGEmbeder', help='model name')
    parser.add_argument('--dataset', type=str, default='dataset', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')
   
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual", action="store_true", default=True, help="Visualize training status in tensorboard")
    # Training Arguments
    parser.add_argument('--log_every', type=int, default=50, help='interval to log autoencoder training results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--val_every', type=int, default=1)

    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # make output directory if it doesn't already exist
    os.makedirs(f'./output/{args.model}/{args.dataset}/models', exist_ok=True)
    os.makedirs(f'./output/{args.model}/{args.dataset}/tmp_results', exist_ok=True)
    
    torch.backends.cudnn.benchmark = True # speed up training by using cudnn
    torch.backends.cudnn.deterministic = True # fix the random seed in cudnn
   
    train(args)
