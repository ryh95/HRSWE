import os
import pickle
import time
from os.path import join

from sklearn.metrics import precision_recall_fscore_support
from web.embedding import Embedding

from constants import VOCAB_DIR, SYN_ANT_CLASSIFY_TASK_DIR, SPECIALIZED_VECS_DIR, \
    ORIGINAL_EMBEDDING, SYN_ANT_CLASSIFY_VECS, ATTRACT_REPEL_VECS, ORIGINAL_VECS_DIR
import numpy as np

from preprocess import GeneralTextProcesser
from syn_ant_classify_task.local_utils import extract_syn_ant_vocab, evaluate_on_task_data, tune_threshold
from utils import  prepare_syn_ant_graph, DAWE

vocab_fname = 'syn_ant_classify_test_val'
fnames_test = ['adjective-pairs.test','noun-pairs.test','verb-pairs.test']
fnames_val = ['adjective-pairs.val','noun-pairs.val','verb-pairs.val']

# Step 1: obtain vocab
if os.path.isfile(join(VOCAB_DIR, vocab_fname + '.npy')):
    words = np.load(join(VOCAB_DIR, vocab_fname + '.npy'))
else:
    vocab = set()
    for f in fnames_test+fnames_val:
        vocab |= extract_syn_ant_vocab(join(SYN_ANT_CLASSIFY_TASK_DIR,'task_data',f))
    words = list(vocab)
    np.save(join(VOCAB_DIR, vocab_fname + '.npy'), words)


# Step 2: extract vectors from original vectors/ specialized vectors
method_name = ''
sel_vec_dir = join(ORIGINAL_VECS_DIR,method_name)
sel_vec_fname = join(sel_vec_dir,vocab_fname)

if os.path.isfile(sel_vec_fname + '.pickle'):
    with open(sel_vec_fname + '.pickle', 'rb') as handle:
        emb_dict = pickle.load(handle)

else:
    text_preprocesser = GeneralTextProcesser()
    emb_dict = text_preprocesser.vocab2vec(words, sel_vec_dir, sel_vec_fname, ORIGINAL_EMBEDDING,
                                           ['pickle'], 'word2vec', normalize=True, oov_handle='mean_emb_vec')


# emb_obj = {}
# with open(join(ATTRACT_REPEL_VECS,'syn_ant_classify_test_sp.txt'),'r') as f:
#     for line in f:
#         word,*vec = line.strip().split(' ')
#         emb_obj[word] = np.array(vec,dtype=float)

# Step 3: test this embedding
print('results of '+ sel_vec_fname + '.pickle')
# first, tune the threshold on dev set

ths = np.linspace(0,1,40)
th,_ = tune_threshold(ths,fnames_val,emb_dict)

# second, test on test set
print('*'*40)
for fname in fnames_test:
    task_fname = join(SYN_ANT_CLASSIFY_TASK_DIR, 'task_data', fname)
    p, r, f1 = evaluate_on_task_data(emb_dict, task_fname, th)
    print('test %s: %f' %(fname.split('-')[0],f1))

# Step 4: specialize this embedding
specialize = True
if specialize:

    print('results of specialized embedding')
    beta1s = np.logspace(-3,2,20)
    beta2s = np.logspace(-3,2,20)
    # betas = [30]
    thesauri = {'name': 'wn_ro'}
    config = {'sim_mat_type': 'pd', 'eig_vec_option': 'ld', 'emb_type': 0}
    cur_best_score = -np.inf
    adj_pos, adj_neg = prepare_syn_ant_graph(words, thesauri)

    words_emb = [emb_dict[w] for w in words]
    words_emb = np.vstack(words_emb).T

    curr_dev_f1 = -np.inf
    times = []
    for beta1 in beta1s:
        for beta2 in beta2s:
            last_time = time.time()
            emb = DAWE(beta1,beta2, words_emb, adj_pos, adj_neg, config)
            time_spend = round(time.time() - last_time, 1)
            times.append(time_spend)
            print('Time took: ', time_spend)

            emb_dict = {w: emb[:, i] for i, w in enumerate(words)}

            # tune threshold first
            th,max_dev_f1 = tune_threshold(ths,fnames_val,emb_dict)
            if max_dev_f1 > curr_dev_f1:
                curr_dev_f1 = max_dev_f1
                best_emb_dict = emb_dict
                best_th = th
                best_beta1 = beta1
                best_beta2 = beta2

    print('Average time spent: ', round(sum(times)/len(times),1))
    # test with this threshold and specialized embedding
    print('best th: %f' % best_th)
    print('best beta1: %f' % best_beta1)
    print('best beta2: %f' % best_beta2)

    for fname in fnames_test:
        task_fname = join(SYN_ANT_CLASSIFY_TASK_DIR, 'task_data', fname)
        p, r, f1 = evaluate_on_task_data(best_emb_dict, task_fname, best_th)
        print('test %s: %f' % (fname.split('-')[0], f1))