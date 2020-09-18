import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score  

parser = argparse.ArgumentParser()
parser.add_argument('--pred', default=None, help='path to pred')
parser.add_argument('--target', default=None, help='path to groundtruth')
args = parser.parse_args()
pred = args.pred
target = args.target

from sklearn.metrics import cohen_kappa_score


y_true = []
y_pred = []
target_names = ['Entailed', 'Contradicted', 'Irrelevant']

with open(pred) as p, open(target) as t:
    for line in p:
        y_pred.append(int(line.strip().split('\t')[1]))
    for line in t:
        y_true.append(int(line.strip().split('\t')[1]))

print(classification_report(y_true, y_pred, target_names=target_names, digits=3))
print(accuracy_score(y_true, y_pred))