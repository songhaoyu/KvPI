import sys
sys.path.append('./lib')
from google_bert import BasicTokenizer
import random
import torch
from treelstm import Tree


class DataLoader:
    def __init__(self, train_path, dev_path, max_len):
        self.tokenizer = BasicTokenizer()
        self.train_path = train_path
        self.dev_path = dev_path
        self.max_len = max_len

        self.train_seg_list, self.train_tgt_list, self.train_segment_list, self.train_type_list, self.train_category_list, self.train_a_seg_list, self.train_a_tree_list, self.train_b_seg_list, self.train_b_tree_list = self.load_data(train_path)
        self.dev_seg_list, self.dev_tgt_list, self.dev_segment_list, self.dev_type_list, self.dev_category_list, self.dev_a_seg_list, self.dev_a_tree_list, self.dev_b_seg_list, self.dev_b_tree_list = self.load_data(dev_path)

        self.train_num, self.dev_num = len(self.train_seg_list), len(self.dev_seg_list)
        print ('train number is %d, dev number is %d' % (self.train_num, self.dev_num))

        num_train_segment, num_dev_segment = len(self.train_segment_list), len(self.dev_segment_list)
        num_train_type, num_dev_type = len(self.train_type_list), len(self.dev_type_list)
        assert num_train_segment == num_train_type == self.train_num
        assert num_dev_segment == num_dev_type == self.dev_num

        self.train_idx_list, self.dev_idx_list = [i for i in range(self.train_num)], [j for j in range(self.dev_num)]
        self.shuffle_train_idx()

        self.train_current_idx = 0
        self.dev_current_idx = 0

    def segment(self, text):
        seg = [1 for _ in range(len(text))]
        idx = text.index("sep")
        seg[:idx] = [0 for _ in range(idx)]
        return [0]+seg+[1]  # [CLS]+seg+[SEP]

    def profile(self, text):
        seg = [3 for _ in range(len(text))]
        loc_idx = text.index("loc")
        gender_idx = text.index("gender")
        sep_idx = text.index("sep")
        seg[:loc_idx] = [0 for _ in range(loc_idx)]
        seg[loc_idx:gender_idx] = [1 for _ in range(gender_idx-loc_idx)]
        seg[gender_idx:sep_idx] = [2 for _ in range(sep_idx-gender_idx)]
        return [0]+seg+[3]  # [CLS]+seg+[SEP]

    def read_trees(self, batch):
        trees = [self.read_tree(line) for line in batch]
        return trees

    def read_tree(self, line):
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

    def load_data(self, path):
        src_list = list()  # src_list contains segmented text
        tgt_list = list()  # tgt_list contains class number
        seg_list = list()  # seg_list contains 0,1 to indicate profile and response
        typ_list = list()  # typ_list contains 0,1,2,3 to indicate constellation, location, gender and response
        cat_list = list()
        a_seg_list = list()
        a_parse_list = list()
        b_seg_list = list()
        b_parse_list = list()
        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                content_list = l.strip('\n').split('\t')
                text = content_list[0]
                target = int(content_list[1])
                category = int(content_list[2])
                a_seg = self.seq_cut(content_list[3].split(' '))
                a_tree = self.read_tree(content_list[4])
                b_seg = self.seq_cut(content_list[5].split(' '))
                b_tree = self.read_tree(content_list[6])
                seg_text = self.tokenizer.tokenize(text)
                post_text = self.seq_cut(seg_text)
                seg_tmp = self.segment(post_text)
                typ_tmp = self.profile(post_text)
                src_list.append(post_text)
                tgt_list.append(target)
                seg_list.append(seg_tmp)
                typ_list.append(typ_tmp)
                cat_list.append(category)

                a_seg_list.append(a_seg)
                a_parse_list.append(a_tree)
                b_seg_list.append(b_seg)
                b_parse_list.append(b_tree)

                assert len(seg_tmp) == len(typ_tmp) == len(post_text)+2

            assert len(src_list) == len(tgt_list) == len(seg_list) == len(typ_list) == len(cat_list)
            assert len(cat_list) == len(a_seg_list) == len(a_parse_list) == len(b_seg_list) == len(b_parse_list)

        return src_list, tgt_list, seg_list, typ_list, cat_list, a_seg_list, a_parse_list, b_seg_list, b_parse_list

    def shuffle_train_idx(self):
        random.shuffle(self.train_idx_list)

    def seq_cut(self, seq):
        if len(seq) > self.max_len:
            seq = seq[ : self.max_len]
        return seq

    def get_next_batch(self, batch_size, mode):
        batch_text_list, batch_label_list = list(), list()
        batch_seg_list, batch_type_list = list(), list()
        batch_category_list = list()
        batch_a_seg_list, batch_a_tree_list = list(), list()
        batch_b_seg_list, batch_b_tree_list = list(), list()
        if mode == 'train':
            if self.train_current_idx + batch_size < self.train_num - 1:
                for i in range(batch_size):
                    curr_idx = self.train_current_idx + i
                    batch_text_list.append(self.train_seg_list[self.train_idx_list[curr_idx]])
                    batch_label_list.append(self.train_tgt_list[self.train_idx_list[curr_idx]])
                    batch_seg_list.append(self.train_segment_list[self.train_idx_list[curr_idx]])
                    batch_type_list.append(self.train_type_list[self.train_idx_list[curr_idx]])
                    batch_category_list.append(self.train_category_list[self.train_idx_list[curr_idx]])
                    batch_a_seg_list.append(self.train_a_seg_list[self.train_idx_list[curr_idx]])
                    batch_a_tree_list.append(self.train_a_tree_list[self.train_idx_list[curr_idx]])
                    batch_b_seg_list.append(self.train_b_seg_list[self.train_idx_list[curr_idx]])
                    batch_b_tree_list.append(self.train_b_tree_list[self.train_idx_list[curr_idx]])
                self.train_current_idx += batch_size
            else:
                for i in range(batch_size):
                    curr_idx = self.train_current_idx + i
                    if curr_idx > self.train_current_idx - 1:
                        self.shuffle_train_idx()
                        curr_idx = 0
                        batch_text_list.append(self.train_seg_list[self.train_idx_list[curr_idx]])
                        batch_label_list.append(self.train_tgt_list[self.train_idx_list[curr_idx]])
                        batch_seg_list.append(self.train_segment_list[self.train_idx_list[curr_idx]])
                        batch_type_list.append(self.train_type_list[self.train_idx_list[curr_idx]])
                        batch_category_list.append(self.train_category_list[self.train_idx_list[curr_idx]])
                        batch_a_seg_list.append(self.train_a_seg_list[self.train_idx_list[curr_idx]])
                        batch_a_tree_list.append(self.train_a_tree_list[self.train_idx_list[curr_idx]])
                        batch_b_seg_list.append(self.train_b_seg_list[self.train_idx_list[curr_idx]])
                        batch_b_tree_list.append(self.train_b_tree_list[self.train_idx_list[curr_idx]])
                    else:
                        batch_text_list.append(self.train_seg_list[self.train_idx_list[curr_idx]])
                        batch_label_list.append(self.train_tgt_list[self.train_idx_list[curr_idx]])
                        batch_seg_list.append(self.train_segment_list[self.train_idx_list[curr_idx]])
                        batch_type_list.append(self.train_type_list[self.train_idx_list[curr_idx]])
                        batch_category_list.append(self.train_category_list[self.train_idx_list[curr_idx]])
                        batch_a_seg_list.append(self.train_a_seg_list[self.train_idx_list[curr_idx]])
                        batch_a_tree_list.append(self.train_a_tree_list[self.train_idx_list[curr_idx]])
                        batch_b_seg_list.append(self.train_b_seg_list[self.train_idx_list[curr_idx]])
                        batch_b_tree_list.append(self.train_b_tree_list[self.train_idx_list[curr_idx]])
                self.train_current_idx = 0

        elif mode == 'dev':
            if self.dev_current_idx + batch_size < self.dev_num - 1:
                for i in range(batch_size):
                    curr_idx = self.dev_current_idx + i
                    batch_text_list.append(self.dev_seg_list[curr_idx])
                    batch_label_list.append(self.dev_tgt_list[curr_idx])
                    batch_seg_list.append(self.dev_segment_list[curr_idx])
                    batch_type_list.append(self.dev_type_list[curr_idx])
                    batch_category_list.append(self.dev_category_list[curr_idx])
                    batch_a_seg_list.append(self.dev_a_seg_list[curr_idx])
                    batch_a_tree_list.append(self.dev_a_tree_list[curr_idx])
                    batch_b_seg_list.append(self.dev_b_seg_list[curr_idx])
                    batch_b_tree_list.append(self.dev_b_tree_list[curr_idx])
                self.dev_current_idx += batch_size
            else:
                for i in range(batch_size):
                    curr_idx = self.dev_current_idx + i
                    if curr_idx > self.dev_num - 1:  # 对dev_current_idx重新赋值
                        curr_idx = 0
                        self.dev_current_idx = 0
                    else:
                        pass
                    batch_text_list.append(self.dev_seg_list[curr_idx])
                    batch_label_list.append(self.dev_tgt_list[curr_idx])
                    batch_seg_list.append(self.dev_segment_list[curr_idx])
                    batch_type_list.append(self.dev_type_list[curr_idx])
                    batch_category_list.append(self.dev_category_list[curr_idx])
                    batch_a_seg_list.append(self.dev_a_seg_list[curr_idx])
                    batch_a_tree_list.append(self.dev_a_tree_list[curr_idx])
                    batch_b_seg_list.append(self.dev_b_seg_list[curr_idx])
                    batch_b_tree_list.append(self.dev_b_tree_list[curr_idx])
                self.dev_current_idx = 0
        else:
            raise Exception('Wrong batch mode!!!')

        return batch_text_list, batch_label_list, batch_seg_list, batch_type_list, batch_category_list, batch_a_seg_list, batch_a_tree_list, batch_b_seg_list, batch_b_tree_list
