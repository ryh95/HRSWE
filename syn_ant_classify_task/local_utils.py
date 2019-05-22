from os.path import join

from sklearn.metrics import precision_recall_fscore_support

from constants import SYN_ANT_CLASSIFY_TASK_DIR
import numpy as np

def extract_syn_ant_vocab(fname):
    vocab = set()
    with open(fname,'r') as f:
        for l in f:
            # 0: synonym 1: antonym
            w1,w2,label = l.strip().split('\t')
            vocab.update([w1,w2])
    return vocab

def evaluate_on_task_data(emb_dict,task_fname,th):
    with open(task_fname,'r') as f:
        labels,y_preds = [],[]
        for l in f:
            w1,w2,label = l.strip().split('\t')
            labels.append(int(label))
            if emb_dict[w1].dot(emb_dict[w2].T)/(np.linalg.norm(emb_dict[w1]) * np.linalg.norm(emb_dict[w2])) > th:
                y_preds.append(0)
            else:
                y_preds.append(1)
    p, r, f1, _ = precision_recall_fscore_support(labels, y_preds, average='binary')
    return p,r,f1

def tune_threshold(thresholds,val_fnames,emb_dict):
    total_f1s = []
    for th in thresholds:
        print('*' * 40)
        print('threshold: %f' %th)
        total_f1 = 0
        for fname in val_fnames:
            task_fname = join(SYN_ANT_CLASSIFY_TASK_DIR, 'task_data', fname)
            p, r, f1 = evaluate_on_task_data(emb_dict, task_fname, th)
            print('dev %s: %f' % (fname.split('-')[0], f1))
            total_f1 += f1
        total_f1s.append(total_f1)

    max_f1 = max(total_f1s)
    best_th = thresholds[total_f1s.index(max_f1)]
    print('*'*40)
    print('best f1 on dev: %f, threshold: %f' %(max_f1,best_th))
    return best_th,max_f1