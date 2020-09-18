import sys
import torch
import argparse
sys.path.append('./lib')
from bert import BERTLM
from treelstm import TreeLSTM
from kvbert import myModel
from kvbert import TreeArgs
from treelstm import treeVocab
import numpy as np
from google_bert import BasicTokenizer
from treelstm import Tree
from treelstm import Constants
from tqdm import tqdm

def extract_parameters(ckpt_path):
    model_ckpt = torch.load(ckpt_path)
    bert_args = model_ckpt['bert_args']
    model_args = model_ckpt['args']
    bert_vocab = model_ckpt['bert_vocab']
    model_parameters = model_ckpt['model']
    tree_args = model_ckpt['tree_args']
    tree_vocab = model_ckpt['tree_vocab']
    return bert_args, model_args, bert_vocab, model_parameters, tree_args, tree_vocab

def init_empty_bert_model(bert_args, bert_vocab, gpu_id, approx = 'none'):
    bert_model = BERTLM(gpu_id, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
            bert_args.dropout, bert_args.layers, approx)
    return bert_model

def init_empty_tree_model(t_args, tree_vocab, gpuid):
    tree_model = TreeLSTM(tree_vocab.size(), t_args.input_dim, t_args.mem_dim, t_args.hidden_dim, t_args.num_classes, t_args.freeze_embed)
    tree_model = tree_model.cuda(gpuid)
    return tree_model


def init_sequence_classification_model(empty_bert_model, args, bert_args, gpu_id, bert_vocab, model_parameters, empty_tree_model, tree_args):
    number_class = args.number_class
    number_category = 3
    embedding_size = bert_args.embed_dim
    batch_size = args.batch_size
    dropout = args.dropout
    tree_hidden_dim = tree_args.hidden_dim
    device = gpu_id
    vocab = bert_vocab
    seq_tagging_model = myModel(empty_bert_model, number_class, number_category, embedding_size, batch_size, dropout, device, vocab, empty_tree_model, tree_hidden_dim)
    seq_tagging_model.load_state_dict(model_parameters)
    return seq_tagging_model

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--test_data',type=str)
    parser.add_argument('--out_path',type=str)
    parser.add_argument('--gpu_id',type=int, default=0)
    return parser.parse_args()

def segment(text):
    seg = [1 for _ in range(len(text))]
    idx = text.index("sep")
    seg[:idx] = [0 for _ in range(idx)]
    return seg

def profile(text):
    seg = [3 for _ in range(len(text))]
    loc_idx = text.index("loc") - 1
    gender_idx = text.index("gender") - 1
    sep_idx = text.index("sep")
    seg[:loc_idx] = [0 for _ in range(loc_idx)]
    seg[loc_idx:gender_idx] = [1 for _ in range(gender_idx-loc_idx)]
    seg[gender_idx:sep_idx] = [2 for _ in range(sep_idx-gender_idx)]
    return seg


def read_tree(line):
    parents = list(map(int, line.split()))
    trees = dict()
    root = None
    for i in range(1, len(parents) + 1):
        if i - 1 not in trees.keys() and parents[i - 1] != -1:
            idx = i
            prev = None
            while True:
                parent = parents[idx - 1]
                if parent == -1:
                    break
                tree = Tree()
                if prev is not None:
                    tree.add_child(prev)
                trees[idx - 1] = tree
                tree.idx = idx - 1
                if parent - 1 in trees.keys():
                    trees[parent - 1].add_child(tree)
                    break
                elif parent == 0:
                    root = tree
                    break
                else:
                    prev = tree
                    idx = parent
    return root


def seq_cut(seq, max_len):
    if len(seq) > max_len:
        seq = seq[:max_len]
    return seq


def read_sentence(line, vocab):
    indices = vocab.convertToIdx(line, Constants.UNK_WORD)
    return torch.LongTensor(indices)


if __name__ == '__main__':
    args = parse_config()
    ckpt_path = args.ckpt_path
    test_data = args.test_data
    out_path = args.out_path
    gpu_id = args.gpu_id

    bert_args, model_args, bert_vocab, model_parameters, tree_args, tree_vocab = extract_parameters(ckpt_path)
    empty_bert_model = init_empty_bert_model(bert_args, bert_vocab, gpu_id, approx='none')
    empty_tree_model = init_empty_tree_model(tree_args, tree_vocab, gpu_id)
    seq_classification_model = init_sequence_classification_model(empty_bert_model, model_args, bert_args, gpu_id, bert_vocab, model_parameters, empty_tree_model, tree_args)
    seq_classification_model.cuda(gpu_id)

    tokenizer = BasicTokenizer()

    seq_classification_model.eval()
    with torch.no_grad():
        with open(out_path, 'w', encoding='utf8') as o:
            with open(test_data, 'r', encoding='utf8') as i:
                lines = i.readlines()
                for l in tqdm(lines, desc='Predicting'):
                    content_list = l.strip().split('\t')
                    text = content_list[0]
                    text_tokenized_list = tokenizer.tokenize(text)
                    if len(text_tokenized_list) > args.max_len:
                        text_tokenized_list = text_tokenized_list[:args.max_len]
                    seg_list = segment(text_tokenized_list)
                    typ_list = profile(text_tokenized_list)

                    a_seg = read_sentence(seq_cut(content_list[3].split(' '), args.max_len), tree_vocab)
                    a_tree = read_tree(content_list[4])
                    b_seg = read_sentence(seq_cut(content_list[5].split(' '), args.max_len), tree_vocab)
                    b_tree = read_tree(content_list[6])

                    pred_output = seq_classification_model([text_tokenized_list], [seg_list], [typ_list], [a_seg], [a_tree], [b_seg], [b_tree], fine_tune=False)[0].cpu().numpy()
                    pred_probability = pred_output[0]
                    pred_label = np.argmax(pred_probability)
                    out_line = text + '\t' + str(pred_label)
                    o.writelines(out_line + '\n')
    print("done.")
