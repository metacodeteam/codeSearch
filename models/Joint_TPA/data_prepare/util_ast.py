import torch
import numpy as np
import argparse
import json
from collections import Counter
import networkx as nx
import dgl
import data_prepare.Constants

PAD_ID, UNK_ID = [0, 1]

def split_ast_data(args):
    index = np.load(args.shuffle_index_file)

    dir_path = args.data_path + args.dataset
    all_ast_file_path = dir_path + args.all_ast_file
    train_ast_file_path = dir_path + args.train_ast_file
    test_ast_file_path = dir_path + args.test_ast_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(all_ast_file_path, 'r') as all_ast_file:
        lines = all_ast_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(lines)])
    print('ast_num:\n', len(mark_list))

    with open(train_ast_file_path, 'w') as train_ast_file,  open(test_ast_file_path, 'w') as test_ast_file:
        for i in range(0, args.trainset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                train_ast_file.write(lines[j])
        for i in range(args.testset_start_index, args.testset_start_index+args.testset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                test_ast_file.write(lines[j])

class multidict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value



def txt2json(ast_txt_file_path):
    mark_list = []
    start_index, end_index = [0, 0]
    ast_cnt = 1
    with open(ast_txt_file_path, 'r') as ast_txt_file:
        ast_lines = ast_txt_file.readlines()
        for i in range(0, len(ast_lines)):
            if ast_lines[i][0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
                ast_cnt += 1
                #print('ast_number:\n', ast_cnt)
        mark_list.append([start_index, len(ast_lines)])

    tree_dict = multidict()
    for i in range(0, ast_cnt):
        start_index, end_index = mark_list[i]
        #print(ast_lines[start_index+1].split()[0])
        try:
            root_idx = int(ast_lines[start_index+1].split()[0]) 
        except ValueError as e: 
            print(ast_lines[start_index+1].split()) 
        tree_dict[i][root_idx]['parent'] = None 
        for j in range(start_index+1, end_index):
            node_list = ast_lines[j].split() 
            f_node, s1_node, s2_node = int(node_list[0]), int(node_list[1]), int(node_list[2])
            if s1_node == -1:
                try:
                    tree_dict[i][f_node]['children'] = [node_list[3]] 
                except IndexError as e: 
                    print('i = {}, f_node = {}, node_list = {}'.format(i, f_node, node_list))
                    tree_dict[i][f_node]['children'] = ['waste'] 
            else:
                tree_dict[i][f_node]['children'] = [s1_node, s2_node]
                tree_dict[i][s1_node]['parent'] = f_node 
                tree_dict[i][s2_node]['parent'] = f_node
    
    tree_dict_str = json.dumps(tree_dict)
    ast_json_file_path = ast_txt_file_path[0:-3] + 'json'
    with open(ast_json_file_path, 'w') as ast_json_file:
        ast_json_file.write(tree_dict_str)


def create_ast_dict_file(args):
    dir_path = args.data_path + args.dataset
    ast_file_path = dir_path + args.train_ast_json_file 

    input_desc = []
    ast_dict = json.loads(open(ast_file_path, 'r').readline())
    ast_words = []
    for i in range(0, len(ast_dict)): 
        tree_dict = ast_dict[str(i)] 
        for node_index in tree_dict.keys(): 
            try:
                if len(tree_dict[node_index]['children']) == 1: 
                    ast_words.append(tree_dict[node_index]['children'][0])
            except KeyError as e:
                print('i = {}, node_index = {}, tree_dict[node_index] = {}'.format(i, node_index, tree_dict[node_index]))
    vocab_ast_info = Counter(ast_words)
    print('word_num:\n', len(vocab_ast_info))

    vocab_ast = [item[0] for item in vocab_ast_info.most_common()[:args.ast_word_num-2]]
    vocab_ast_index = {'<pad>':0, '<unk>':1}
    vocab_ast_index.update(zip(vocab_ast, [item+2 for item in range(len(vocab_ast))]))


    vocab_ast_file_path = dir_path + args.vocab_ast_file
    ast_dic_str = json.dumps(vocab_ast_index)
    with open(vocab_ast_file_path, 'w') as vocab_ast_file:
        vocab_ast_file.write(ast_dic_str)


def build_tree_useless(tree_json, ast_dict):
    g = nx.DiGraph()

    def _rec_build(node):
        children = [c for c in node['children']]
        word_id = ast_dict.get(node['value'],UNK_ID)
        g.add_node(int(node['id']),x=word_id, y=node['id'], mask=1)
        if len(children) == 2:
            g.add_edge(children[0],node['id'])
            g.add_edge(children[1],node['id'])
        elif len(children) == 1:
            g.add_edge(children[0],node['id'])

    for node in tree_json:
        _rec_build(node)
    ret = dgl.DGLGraph()
    ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
    # ret: dgl.DGLGraph()
    return ret


def build_tree_not_use(tree_json, ast_dict):
    g = nx.DiGraph()

    def _rec_build(nid, idx, t_json):
        for i, node in t_json.items():
            if node['value'] == 'Tmp':
                ori_number = i
                break

        children = [c for c in t_json[idx]['children']]

        if len(children) == 2:

            if nid is None:
                g.add_node(0, x=-1, y=int(idx), mask=0)
                nid = 0

            for c in children:
                cid = g.number_of_nodes()

                y_value = c
                c = str(c)

                c_children = t_json[c]['children']
                c_children_list = [c_tmp for c_tmp in c_children if c_tmp<ori_number]


                if len(c_children_list) == 2:
                    g.add_node(cid, x=-1, y=y_value, mask=0)
                    _rec_build(cid, c, t_json)

                else:
                    assert len(t_json[c]['children']) == 1
                    word_index = ast_dict.get(t_json[c]['children'][0], UNK_ID)
                    g.add_node(cid, x=word_index, y=y_value, mask=1)

                g.add_edge(cid, nid)
                # print('father:{}, son:{}'.format(g.nodes[nid]['y'], g.nodes[cid]['y']))

        else:
            assert len(t_json[idx]['children']) == 1
            word_index = ast_dict.get(t_json[idx]['children'][0], UNK_ID)
            if nid is None:
                cid = 0
            else:
                cid = g.number_of_nodes()

            g.add_node(cid, x=word_index, y=int(idx), mask=1)

            if nid is not None:
                g.add_edge(cid, nid)
                # print('father:{}, son:{}'.format(g.nodes[nid]['y'], g.nodes[cid]['y']))

    # for k, node in tree_json.items():
    # if node['parent'] == None:
    # root_idx = k
    root_idx = str(0)
    _rec_build(None, root_idx, tree_json)
    ret = dgl.DGLGraph()
    ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])

    # ret: dgl.DGLGraph()
    return ret
def build_tree(tree_json, dict_code):
    tmp_nodename_list = []
    nodename_int_list = []
    for nodename in tree_json.keys():
        if nodename not in tmp_nodename_list:
            tmp_nodename_list.append(nodename)
            nodename_int_list.append(int(nodename[len(data_prepare.Constants.NODE_FIX):]))

    assert len(tree_json) == len(tmp_nodename_list)

    g = nx.DiGraph()

    def _rec_build(nid, idx, t_json):

        children = [c for c in t_json[idx]['children'] if c.startswith(data_prepare.Constants.NODE_FIX)]

        if len(children) == 2:

            if nid is None:
                g.add_node(0, x=data_prepare.Constants.DGLGraph_PAD_WORD, y=int(idx[len(data_prepare.Constants.NODE_FIX):]), mask=0)
                nid = 0

            for c in children:
                cid = g.number_of_nodes()

                y_value = int(c[len(data_prepare.Constants.NODE_FIX):])

                c_children = t_json[c]["children"]
                c_children_list = [c_tmp for c_tmp in c_children if c_tmp.startswith(data_prepare.Constants.NODE_FIX)]

                if len(c_children_list) == 2:
                    g.add_node(cid, x=data_prepare.Constants.DGLGraph_PAD_WORD, y=y_value, mask=0)

                    _rec_build(cid, c, t_json)
                else:
                    assert len(t_json[c]['children']) == 1
                    word_index = dict_code.get(t_json[c]['children'][0], data_prepare.Constants.UNK)
                    g.add_node(cid, x=word_index, y=y_value, mask=1)

                g.add_edge(cid, nid)
        else:
            assert len(t_json[idx]['children']) == 1
            word_index = dict_code.get(t_json[idx]['children'][0], data_prepare.Constants.UNK)
            if nid is None:
                cid = 0
            else:
                cid = g.number_of_nodes()
            y_value = int(idx[len(data_prepare.Constants.NODE_FIX):])
            g.add_node(cid, x=word_index, y=y_value, mask=1)

            if nid is not None:
                g.add_edge(cid, nid)

    for k, node in tree_json.items():
        if node['parent'] == None:
            root_idx = k

    _rec_build(None, root_idx, tree_json)
    ret = dgl.DGLGraph()

    # nx_nodename_int_list = []
    # for nx_node_id in g.nodes():
    #     if g.node[nx_node_id]["y"] not in nx_nodename_int_list:
    #         nx_nodename_int_list.append(g.node[nx_node_id]["y"])
    #
    # nodename_int_not_in_nx = []
    # for this_node_name_int in nodename_int_list:
    #     if this_node_name_int not in nx_nodename_int_list:
    #         nodename_int_not_in_nx.append(this_node_name_int)

    ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
    return ret


def parse_args():
    parser = argparse.ArgumentParser("Parse AST data for ASTEmbedder")
    
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='dataset/')

    parser.add_argument('--all_ast_file', type=str, default='all.ast.txt')
    parser.add_argument('--train_ast_file', type=str, default='train.ast.txt')
    parser.add_argument('--test_ast_file', type=str, default='test.ast.txt')
    parser.add_argument('--train_ast_json_file', type=str, default='train.ast.json')
    parser.add_argument('--test_ast_json_file', type=str, default='test.ast.json')

    parser.add_argument('--vocab_ast_file', type=str, default='vocab.ast.json')
    
    parser.add_argument('--trainset_num', type=int, default=32000)
    parser.add_argument('--testset_num', type=int, default=1000)
    parser.add_argument('--ast_word_num', type=int, default=5000)
    parser.add_argument('--testset_start_index', type=int, default=33000)


    parser.add_argument('--shuffle_index_file', type=str, default='shuffle_index.npy')
 
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    #split_data(args)
    
    dir_path = args.data_path + args.dataset
    ast_txt_file_path = dir_path + args.test_ast_file
    txt2json(ast_txt_file_path)
    
    #create_dict_file(args)
    
    mark_list = []
    start_index, end_index = [0, 0]
    ast_cnt = 1
    dir_path = args.data_path + args.dataset
    ast_txt_file_path = dir_path + args.train_ast_file
    with open(ast_txt_file_path, 'r') as ast_txt_file:
        ast_lines = ast_txt_file.readlines()
        for i in range(0, len(ast_lines)):
            if ast_lines[i][0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
                ast_cnt += 1
                #print('ast_number:\n', ast_cnt)
        mark_list.append([start_index, len(ast_lines)])
    #print(mark_list[400])
    
    dir_path = args.data_path + args.dataset
    ast_file_path = dir_path + args.train_ast_json_file
    ast_dict_path = dir_path + args.vocab_ast_file
    ast_tree_json = json.loads(open(ast_file_path, 'r').readline())
    vacab_ast_dict = json.loads(open(ast_dict_path, "r").readline())


    tree_json = ast_tree_json[str(0)]
    ret = build_tree(tree_json, vacab_ast_dict)
    
