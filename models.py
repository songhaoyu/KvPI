import sys
sys.path.append('./lib')
import torch
from torch import nn
import torch.nn.functional as F
import random
import os
import argparse

#### Load pretrained bert model
from bert import BERTLM
from data import Vocab, CLS, SEP, MASK
from data_loader import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from treelstm import Constants
from treelstm import TreeLSTM
from treelstm import treeVocab
from treelstm import utils
from treelstm import Tree


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str)
    parser.add_argument('--tree_path', type=str)
    parser.add_argument('--bert_vocab', type=str)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--number_class', type = int)
    parser.add_argument('--number_category', type=int)
    parser.add_argument('--number_epoch', type = int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--model_save_path', type=str)
    return parser.parse_args()


class TreeArgs(object):
    def __init__(self):
        super(TreeArgs, self).__init__()
        self.input_dim = 300
        self.mem_dim = 150
        self.hidden_dim = 50
        self.num_classes = 3
        self.freeze_embed = False
        self.epochs = 15
        self.batchsize = 25
        self.lr = 0.01
        self.wd = 1e-4
        self.optim = 'adagrad'
        self.seed = 8273
        self.cuda = True


def init_bert_model(args, device, bert_vocab):
    bert_ckpt = torch.load(args.bert_path, map_location='cpu')
    bert_args = bert_ckpt['args']
    bert_vocab = Vocab(bert_vocab, min_occur_cnt=bert_args.min_occur_cnt, specials=[CLS, SEP, MASK])
    bert_model = BERTLM(device, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, bert_args.dropout, bert_args.layers, bert_args.approx)
    pretrained_dict = bert_ckpt['model']
    model_dict = bert_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # difference = {k: v for k, v in model_dict.items() if k not in pretrained_dict}
    model_dict.update(pretrained_dict)
    bert_model.load_state_dict(model_dict)
    # bert_model.load_state_dict(bert_ckpt['model'])
    nn.init.normal_(bert_model.state_dict()['typ_embed.weight'], std=0.02)
    bert_model = bert_model.cuda(device)
    return bert_model, bert_vocab, bert_args


def init_tree_model(args, device, t_vocab):
    t_ckpt = torch.load(args.tree_path, map_location='cpu')
    t_args = t_ckpt['tree_args']
    t_model = TreeLSTM(t_vocab.size(), t_args.input_dim, t_args.mem_dim, t_args.hidden_dim, t_args.num_classes, t_args.freeze_embed)
    pretrained_dict = t_ckpt['tree_model']
    model_dict = t_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    t_model.load_state_dict(model_dict)
    t_model = t_model.cuda(device)
    for para in t_model.parameters():
        para.requires_grad = False
    return t_model, t_args


def ListsToTensor(xs, seg, typ, vocab):
    batch_size = len(xs)
    lens = [ len(x)+2 for x in xs]
    mx_len = max(lens)
    ys = []
    seg_padded = []
    typ_padded = []
    for i, x in enumerate(xs):
        y = vocab.token2idx([CLS]+x+[SEP]) + ([vocab.padding_idx]*(mx_len - lens[i]))
        ys.append(y)
    for i, x in enumerate(seg):
        seg_padded.append(x + [1 for _ in range(mx_len-len(x))])
    for i, x in enumerate(typ):
        typ_padded.append(x + [3 for _ in range(mx_len-len(x))])
    data = torch.LongTensor(ys).t_().contiguous()
    seg_tsr = torch.LongTensor(seg_padded).t_().contiguous()
    typ_tsr = torch.LongTensor(typ_padded).t_().contiguous()
    return data, seg_tsr, typ_tsr


def batchify(data, seg, typ, vocab):
    return ListsToTensor(data, seg, typ, vocab)


def read_sentences(batch, vocab):
    sentences = [read_sentence(line, vocab) for line in batch]
    return sentences


def read_sentence(line, vocab):
    indices = vocab.convertToIdx(line, Constants.UNK_WORD)
    return torch.LongTensor(indices)


def tree_batchify(linput_txt, rinput_txt, vocab):
    linpu = read_sentences(linput_txt, vocab)
    rinpu = read_sentences(rinput_txt, vocab)
    return linpu, rinpu


class myModel(nn.Module):
    def __init__(self, bert_model, num_class, num_category, embedding_size, batch_size, dropout, device, vocab, tree_model, tree_hidden_dim):
        super(myModel, self).__init__()
        self.bert_model = bert_model
        self.tree_model = tree_model
        self.tree_hidden_dim = tree_hidden_dim
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.num_category = num_category
        self.vocab = vocab
        self.fc = nn.Linear(self.embedding_size + self.tree_hidden_dim, self.num_class)
        # self.fc_2 = nn.Linear(self.embedding_size + self.tree_hidden_dim, self.num_category)
    
    def forward(self, data, seg, typ, ainput, atree, binput, btree, fine_tune=False):
        # size of data is [batch_max_len, batch_size]
        batch_size = len(data)
        data, seg_data, typ_data = batchify(data, seg, typ, self.vocab)
        data = data.cuda(self.device)
        seg_data = seg_data.cuda(self.device)
        typ_data = typ_data.cuda(self.device)
        x = self.bert_model.work(data, seg=seg_data, typ=typ_data)[1].cuda(self.device)
        if not fine_tune:
            x = x.detach()

        x_2 = torch.zeros(batch_size, self.tree_hidden_dim).cuda(self.device)
        for i in range(batch_size):
            li, lt = ainput[i].cuda(self.device), atree[i]
            ri, rt = binput[i].cuda(self.device), btree[i]
            _, x_t = self.tree_model(lt, li, rt, ri)
            x_2[i] = x_t.cuda(self.device)

        x = x.view(batch_size, self.embedding_size)
        x = torch.cat((x, x_2), dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        c1 = self.fc(x)
        # c2 = self.fc(x)
        return c1, c1
