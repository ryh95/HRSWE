import os
import random
import sys
from collections import defaultdict
from os.path import join

import networkx as nx
from scipy import linalg
from six import iteritems
import numpy as np

from sklearn.utils import Bunch

from constants import THESAURUS_DIR, AR_THES_DIR
from posdef import nearestPD


def prepare_syn_ant_graph(words, thesauri):
    G = nx.Graph()
    G.add_nodes_from(words)
    words_set = set(words)
    if 'syn_fname' not in thesauri or 'ant_fname' not in thesauri:
        # ref: https://stackoverflow.com/a/2950027
        sys.exit('No thesauri found!')
    # if thesauri['name'] == 'word_sim':
    #     data = thesauri['word_sim_pairs']
    #     for (w1, w2), y in zip(data.X, data.y):
    #         if y > 8:
    #             G.add_edge(w1, w2, weight=1)
    #         elif y < 2:
    #             G.add_edge(w1, w2, weight=-1)
    syn_constraints = set()
    with open(thesauri['syn_fname'], 'r') as f:
        for line in f:
            word_pair = line.split()
            word_pair = [word[3:] for word in word_pair]  # remove the 'en-' prefix
            if word_pair[0] in words_set and word_pair[1] in words_set and word_pair[0] != word_pair[1]:
                syn_constraints |= {(word_pair[0], word_pair[1])}

    ant_constraints = set()
    with open(thesauri['ant_fname'], 'r') as f:
        for line in f:
            word_pair = line.split()
            word_pair = [word[3:] for word in word_pair]  # remove the 'en-' prefix
            if word_pair[0] in words_set and word_pair[1] in words_set and word_pair[0] != word_pair[1]:
                ant_constraints |= {(word_pair[0], word_pair[1])}

    # Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
    # to be consistent with attract_repel's processing
    for antonym_pair in ant_constraints:
        if antonym_pair in syn_constraints:
            syn_constraints.remove(antonym_pair)

    positive_edges = ((k, v, 1) for k,v in syn_constraints)
    negative_edges = ((k, v, -1) for k,v in ant_constraints)
    G.add_weighted_edges_from(positive_edges)
    # If a pair of words has positive edge and negative edge, the positive edge will be removed
    # then the previous post-processing step can be removed
    G.add_weighted_edges_from(negative_edges)

    adj = nx.adjacency_matrix(G, nodelist=words)

    adj_pos = adj.copy()
    adj_pos[adj < 0] = 0
    adj_pos.eliminate_zeros()

    adj_neg = adj.copy()
    adj_neg[adj > 0] = 0
    adj_neg.eliminate_zeros()
    return adj_pos, adj_neg

def is_psd(A):
    # ref: https://stackoverflow.com/a/16270026
    # TODO: since there are numerical round offs, try other methods to check PSD
    # some useful links to help checking PSD:
    # https: // www.mathworks.com / matlabcentral / answers / 84287 - positive - semi - definite - matrix - problem
    # https: // www.mathworks.com / matlabcentral / answers / 366140 - eig - gives - a - negative - eigenvalue -
    # for -a - positive - semi - definite - matrix
    return np.all(linalg.eigvals(A)>=0)


def combine_bunches(*bunches):
    com_dict = defaultdict(list)
    for bunch in bunches:
        for k,v in bunch.items():
            com_dict[k].append(v)
    X_com=np.vstack(com_dict['X'])
    y_com=np.hstack([y.squeeze() for y in com_dict['y']])
    return Bunch(X=X_com,y=y_com)

def get_words_vecs_from_word_sim(tasks,emb_dict):
    words = set()
    for name, data in iteritems(tasks):
        words |= set(data.X.flatten())
    words = list(words)
    words_vecs = np.vstack([emb_dict[word] for word in words])
    return words, words_vecs


def DAWE(beta1,beta2,emb,adj_pos,adj_neg,config):
    """
    Dynamic Adjusted Word Embedding

    :param float beta:
    :param list words:
    :param nxd ndarray emb:
    :param dict thesauri: keys, [name, word_sim_pairs(if you use word sim data to create graph)]
    :param dict config:  keys, [sim_mat_type(pd/n, n means normal), eig_vec_option(ld, largest eig value),
    emb_type]

    :return: dxn ndarray new word embedding
    """

    d,n = emb.shape
    W = emb.T @ emb

    W_prime = W + beta1 * adj_pos.multiply(np.max(W) - W) + beta2 * adj_neg.multiply(W - np.min(W))

    if config['sim_mat_type'] == 'pd':
        W_hat = nearestPD(W_prime)
    elif config['sim_mat_type'] == 'n':
        W_hat = W_prime

    if config['eig_vec_option'] == 'ld':
        # choose d largest eigenvectors
        # ref: https://stackoverflow.com/a/12168664
        lamb_s, Q_s = linalg.eigh(W_hat, eigvals=(n - d, n - 1))

    if config['emb_type'] == 0: # together use with pd
        new_emb = Q_s @ np.diag(lamb_s ** (1 / 2))
    elif config['emb_type'] == 1: # together use with n
        row_norm = np.linalg.norm(Q_s,axis=1)[:,np.newaxis]
        new_emb = Q_s / row_norm
    # elif config['emb_type'] == 2: # together use with n
    #     assert np.all(lamb_s) >= 0
    #     new_emb = Q_s @ np.diag(lamb_s ** (1 / 2))
    return new_emb.T


def test_on_gre(fname,emb_obj):
    with open(fname,'r') as f:
        answer,correct,total = 0,0,0
        for l in f:
            total += 1
            tc_words,ans_word = l.strip().split('::')
            ans_word = ans_word.strip()
            t_word,c_words = tc_words.split(':')
            c_words = c_words.strip().split(' ')
            t_emb = emb_obj[t_word]

            sims = [t_emb.dot(emb_obj[c_word].T) for c_word in c_words]
            # ref: https: // stackoverflow.com / a / 11825864
            min_index = min(range(len(sims)), key=sims.__getitem__)
            if c_words[min_index] == ans_word:
                correct += 1
            answer += 1
    p = correct / answer
    r = correct / total
    f1 = 2*p*r/(p+r)
    return p,r,f1


def select_syn_ant_sample(syn_ant_fname,out_fname,vocab):
    '''
    select a subset of thesaurus
    :param syn_ant_fname:
    :param out_fname:
    :param vocab: set
    :return:
    '''
    with open(syn_ant_fname,'r') as f,\
         open(out_fname,'w') as f_out:
        for line in f:
            w1,w2 = line.strip().split()
            if w1[3:] in vocab and w2[3:] in vocab:
                f_out.write(w1+' '+w2+'\n')

def adv_attack_thesaurus(syn_fname,ant_fname,po_ratio=0.5,reverse=False):
    syn_pairs,ant_pairs = set(),set()
    with open(syn_fname,'r') as f_syn:
        for line in f_syn:
            syn_pairs.add(line)
    with open(ant_fname,'r') as f_ant:
        for line in f_ant:
            ant_pairs.add(line)

    # random choose po_ratio syn pairs and put them to ant pairs


    if reverse:
        # choose ant pairs and put into syn pairs
        n_choose = int(len(ant_pairs) * po_ratio)
        chosen_pairs = set(random.sample(ant_pairs, n_choose))
        syn_pairs |= chosen_pairs
        ant_pairs -= chosen_pairs
    else:
        # choose syn pairs and put into ant pairs
        n_choose = int(len(syn_pairs) * po_ratio)
        chosen_pairs = set(random.sample(syn_pairs, n_choose))
        syn_pairs -= chosen_pairs
        ant_pairs |= chosen_pairs

    head,syn_name = os.path.split(syn_fname)
    syn_out_fname = join(head,'adv_'+syn_name)
    head,ant_name = os.path.split(ant_fname)
    ant_out_fname = join(head,'adv_'+ant_name)
    with open(syn_out_fname,'w') as f_syn_out:
        for line in syn_pairs:
            f_syn_out.write(line)
    with open(ant_out_fname, 'w') as f_ant_out:
        for line in ant_pairs:
            f_ant_out.write(line)
