import numpy as np
import argparse

cfg_node_color2index = {"black": 0, "yellow": 1, "red": 2, "green": 3, "blue": 4}
cfg_edge_color2index = {"black": 0, "red": 1}


def get_cfg_npy_info(lines, n_node, n_edge_types, state_dim, annotation_dim):
    
    all_num = 0 # count the number of cfgs
    for i in range(0, len(lines)):
        line = lines[i]
        if (line[0:10] == 'BeginFunc:'):
            all_num += 1
    #print('number of cfgs:\n', all_num)

    all_adjmat = np.zeros([all_num, n_node, n_node * n_edge_types * 2])
    #all_adjmat = np.zeros([all_num, n_node, n_node * n_edge_types * 2], dtype='float16')
    all_anno = np.zeros([all_num, n_node, annotation_dim])
    all_node_mask = np.zeros([all_num, n_node])
    cnt = 0
    for i in range(0, len(lines)):
        line = lines[i]

        if (line[0:10] == 'BeginFunc:'): 

            cfg_info_list = lines[i+1].split() # node_num and edge_num of current cfg
            node_num, edge_num = int(cfg_info_list[0]), int(cfg_info_list[1])

            save_node_feature_dict, save_edge_digit_list = {}, []
            for j in range(i+2, i+2+edge_num):
                start_node_info, edge_type, end_node_info = lines[j].split()
                start_node, start_node_feature = start_node_info.split(':')
                end_node, end_node_feature = end_node_info.split(':')   

                reset_edge_type = int(edge_type) 
                if reset_edge_type == 2:
                    reset_edge_type = 0

                
                if int(start_node) < n_node and int(end_node) < n_node:
                    save_edge_digit_list.append([int(start_node), reset_edge_type, int(end_node)])
                if int(start_node) < n_node:
                    save_node_feature_dict[int(start_node)] = int(start_node_feature)
                if int(end_node) < n_node:
                    save_node_feature_dict[int(end_node)] = int(end_node_feature)

            # adjmat: [n_node x (n_node * n_edge_types * 2)]
            adjmat = create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types)
            # anno: [n_node x annotation_dim]
            anno = create_annotation_matrix(save_node_feature_dict, n_node, annotation_dim)
            # node_mask: [n_node]
            node_mask = [1 if k < node_num else 0 for k in range(n_node)]

            all_adjmat[cnt, :, :] = adjmat
            all_anno[cnt, :, :] = anno
            all_node_mask[cnt, :] = node_mask

            cnt += 1
            i += (edge_num + 1)
            '''
            if (cnt == 1):
                print('adjmat.size:\n', adjmat.shape)
                for i in range(0, len(adjmat)):
                    line = adjmat[i]
                    for j in range(0, len(line)):
                        if adjmat[i][j] == 1:
                            print('i:{}, j:{}'.format(i, j))
                print('anno.size:\n', anno.shape)
                for i in range(0, len(anno)):
                    line = anno[i]
                    for j in range(0, len(line)):
                        if anno[i][j] == 1:
                            print('i:{}, j:{}'.format(i, j))
                print('node_mask:\n', len(node_mask))
                for i in range(0, len(node_mask)):
                    if node_mask[i] == 1:
                        print(i)
            '''
    all_init_input = pad_anno(all_anno, n_node, state_dim, annotation_dim)
    # all_adjmat: [all_num x n_node x (n_node * n_edge_types * 2)]
    # all_init_input: [all_num x n_node x state_dim]
    # all_node_mask: [all_num x n_node]
    #print('type of adjmat:\n', type(adjmat))

    return all_init_input, all_adjmat, all_node_mask
#
# def get_one_cfg_npy_info(json_cfg_dict, n_node, n_edge_types, state_dim, max_word_num):
#     node_num = min(len(json_cfg_dict), n_node)
#     save_node_feature_dict, save_edge_digit_list = {}, []
#     anno = np.zeros([n_node, max_word_num])
#
#     for i in range(0, node_num):
#         word_list = json_cfg_dict[str(i)]['wordid']
#         word_num_this_node = len(word_list)
#
#         # 将循环方式改为只取前max个单词
#         # for j in range(0, word_num_this_node):
#         for j in range(0, min(max_word_num, word_num_this_node)):
#             try:
#                 anno[i][j] = word_list[j]
#             except Exception as e:
#                 print(e)
#
#         if 'snode_cfg' in json_cfg_dict[str(i)].keys():
#             snode_list = json_cfg_dict[str(i)]['snode_cfg']
#             for k in range(0, min(n_node, len(snode_list))):
#                 snode = snode_list[k]
#                 if snode < n_node:
#                     save_edge_digit_list.append([i, snode, 0])
#
#     adjmat = create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types)
#     node_mask = [1 if k < node_num else 0 for k in range(0, n_node)]
#     init_input = pad_one_anno(anno, n_node, state_dim, max_word_num)
#
#     return adjmat, init_input, node_mask
# # 使用cfg——att.util_cfg中的同名方法
# """def get_one_cfg_npy_info(lines, n_node, n_edge_types, state_dim, annotation_dim):
#     assert(lines[0][0:10] == 'BeginFunc:')
#
#     cfg_info_list = lines[1].split() # node_num and edge_num of current cfg
#     node_num, edge_num = int(cfg_info_list[0]), int(cfg_info_list[1])
#
#     save_node_feature_dict, save_edge_digit_list = {}, []
#     for j in range(2, 2+edge_num):
#         start_node_info, edge_type, end_node_info = lines[j].split()
#         start_node, start_node_feature = start_node_info.split(':')
#         end_node, end_node_feature = end_node_info.split(':')
#
#         reset_edge_type = int(edge_type)
#         if reset_edge_type == 2:
#             reset_edge_type = 0
#
#
#         if int(start_node) < n_node and int(end_node) < n_node:
#             save_edge_digit_list.append([int(start_node), reset_edge_type, int(end_node)])
#         if int(start_node) < n_node:
#             save_node_feature_dict[int(start_node)] = int(start_node_feature)
#         if int(end_node) < n_node:
#             save_node_feature_dict[int(end_node)] = int(end_node_feature)
#
#     # adjmat: [n_node x (n_node * n_edge_types * 2)]
#     adjmat = create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types)
#     # anno: [n_node x annotation_dim]
#     anno = create_annotation_matrix(save_node_feature_dict, n_node, annotation_dim)
#     init_input = pad_one_anno(anno, n_node, state_dim, annotation_dim)
#     # node_mask: [n_node]
#     node_mask = [1 if k < node_num else 0 for k in range(n_node)]
#     #print('type of adjmat:\n', type(adjmat))
#     return init_input, adjmat, node_mask
# """
# # 使用cfg——att.util_cfg中的同名方法
#
# # def create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types):
# #     a = np.zeros([n_node, n_node * n_edge_types * 2])
# #
# #     for edge in save_edge_digit_list:
# #         src_idx = edge[0]
# #         e_type = edge[1]
# #         tgt_idx = edge[2]
# #
# #         a[tgt_idx][(e_type) * n_node + src_idx] = 1
# #         a[src_idx][(e_type + n_edge_types) * n_node + tgt_idx] = 1
# #
# #     return a
#
# def create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types):
#     a = np.zeros([n_node, n_node * n_edge_types * 2])
#
#     for edge in save_edge_digit_list:
#         src_idx = edge[0]
#         tgt_idx = edge[1]
#         e_type = edge[2]
#
#         a[tgt_idx][(e_type) * n_node + src_idx] = 1
#         a[src_idx][(e_type + n_edge_types) * n_node + tgt_idx] = 1
#
#     return a


def get_one_cfg_npy_info(json_cfg_dict, n_node, n_edge_types, max_word_num):
    node_num = min(len(json_cfg_dict), n_node)
    save_edge_digit_list = []
    anno = np.zeros([n_node, max_word_num])

    for i in range(0, node_num):
        word_list = json_cfg_dict[str(i)]['wordid']
        word_num_this_node = len(word_list)

        for j in range(0, min(word_num_this_node, max_word_num)):
            anno[i][j] = word_list[j]

        if 'snode_cfg' in json_cfg_dict[str(i)].keys():
            snode_list = json_cfg_dict[str(i)]['snode_cfg']
            for j in range(0, len(snode_list)):
                if snode_list[j] < n_node:
                    snode = snode_list[j]
                    save_edge_digit_list.append([i, snode, 0])

    adjmat = create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types)

    node_mask = [1 if k < node_num else 0 for k in range(0, n_node)]

    return anno, adjmat, node_mask


def create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types):
    a = np.zeros([n_node, n_node * n_edge_types * 2])

    for edge in save_edge_digit_list:
        src_idx = edge[0]
        tgt_idx = edge[1]
        e_type = edge[2]

        a[tgt_idx][(e_type) * n_node + src_idx] = 1
        a[src_idx][(e_type + n_edge_types) * n_node + tgt_idx] = 1

    return a


def create_annotation_matrix(save_node_feature_dict, n_node, annotation_dim):
    anno = np.zeros([n_node, annotation_dim])
    for node, node_feature in save_node_feature_dict.items():
        anno[node][node_feature] = 1

    return anno


def pad_anno(all_anno, n_node, state_dim, annotation_dim):
    padding = np.zeros((len(all_anno), n_node, state_dim - annotation_dim))
    all_init_input = np.concatenate((all_anno, padding), 2)
    return all_init_input

def pad_one_anno(anno, n_node, state_dim, annotation_dim):
    padding = np.zeros([n_node, (state_dim - annotation_dim)])
    init_input = np.concatenate((anno, padding), 1)
    return init_input


def split_cfg_data(args):
    index = np.load(args.shuffle_index_file)

    dir_path = args.data_path + args.dataset
    all_cfg_file_path = dir_path + args.all_cfg_file
    train_cfg_file_path = dir_path + args.train_cfg_file
    test_cfg_file_path = dir_path + args.test_cfg_file

    mark_list = []
    cnt = -1
    start_index = 0
    end_index = 0
    with open(all_cfg_file_path, 'r') as all_cfg_file:
        lines = all_cfg_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(lines)])
    #print(mark_list)

    with open(train_cfg_file_path, 'w') as train_cfg_file,  open(test_cfg_file_path, 'w') as test_cfg_file:
        for i in range(0, args.trainset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                train_cfg_file.write(lines[j])
        for i in range(args.testset_start_index, args.testset_start_index+args.testset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                test_cfg_file.write(lines[j])

'''
def parse_args():
    parser = argparse.ArgumentParser("Prepare CFG data for CFGEmbeder")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='github/', help='name of dataset.c')
    
    parser.add_argument('--all_cfg_file', type=str, default='all.cfg.txt')
    parser.add_argument('--train_cfg_file', type=str, default='train.cfg.txt')
    parser.add_argument('--test_cfg_file', type=str, default='test.cfg.txt')  

    parser.add_argument('--n_node', type=int, default=512)
    parser.add_argument('--n_edge_types', type=int, default=2)
    parser.add_argument('--state_dim', type=int, default=512)
    parser.add_argument('--annotation_dim', type=int, default=5)

    parser.add_argument('--trainset_num', type=int, default=32000)
    parser.add_argument('--testset_num', type=int, default=1000)
    parser.add_argument('--testset_start_index', type=int, default=33000)


    parser.add_argument('--shuffle_index_file', type=str, default='shuffle_index.npy')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    #split_data(args)
'''
