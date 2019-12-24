import os
import random
from collections import defaultdict
from os import path
from os.path import join

import numpy
from scipy import linalg
from six import iteritems
import numpy as np
from numpy import linalg as la
from sklearn.utils import Bunch

# ref: https://stackoverflow.com/a/43244194
# another useful ref: https://stackoverflow.com/a/10940283 for finding nearest PSD
# @jit
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3
    # try:
    #     _ = la.cholesky(A3)
    #     return A3
    # except la.LinAlgError:
    #     pass

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    # is_pd = False
    while not isPD(A3):
    # while not is_pd:
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
        # try:
        #     _ = la.cholesky(A3)
        #     is_pd = True
        # except la.LinAlgError:
        #     pass
    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

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


def generate_sub_thesauri(in_fthesauri, out_fthesauri, vocab):
    '''
    select a subset of thesaurus that all thesauri words are in vocab
    :param in_fthesauri:
    :param out_fname:
    :param vocab: set
    :return:
    '''
    if os.path.isfile(out_fthesauri):
        print('sub thesauri has been generated! Quit')
    else:
        with open(in_fthesauri, 'r') as f,\
             open(out_fthesauri,'w') as f_out:
            for line in f:
                w1,w2 = line.strip().split()
                if w1[3:] in vocab and w2[3:] in vocab:
                    f_out.write(w1+' '+w2+'\n')

def get_all_words_in_constraint(word_sim_fname):
    vocab = set()
    with open(word_sim_fname, 'r') as f:
        for line in f:
            word_pair = line.split()
            word_pair = [word[3:] for word in word_pair]  # remove the 'en-' prefix
            vocab.update(word_pair)
    return vocab


def generate_adv1(thesauri, po_ratio=0.5, reverse=False):
    '''
    random choose some positive tuples and put them into negative ones or reverse
    (all tuples are from constrain set used to specialize the embedding)
    :param thesauri: dict, syn_fname/ant_fname : names of files that contain constrains used to specialize the embedding
    :param po_ratio: number of positive(syn) tuples chosen / number of all syn tuples
    :param reverse:
    :return:
    '''
    syn_pairs, ant_pairs = set(), set()
    with open(thesauri['syn_fname'], 'r') as f_syn:
        for line in f_syn:
            syn_pairs.add(line)
    with open(thesauri['ant_fname'], 'r') as f_ant:
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

    head, syn_name = os.path.split(thesauri['syn_fname'])
    syn_out_fname = join(head, 'adv_' + syn_name)
    head, ant_name = os.path.split(thesauri['ant_fname'])
    ant_out_fname = join(head, 'adv_' + ant_name)
    with open(syn_out_fname, 'w') as f_syn_out:
        for line in syn_pairs:
            f_syn_out.write(line)
    with open(ant_out_fname, 'w') as f_ant_out:
        for line in ant_pairs:
            f_ant_out.write(line)


def generate_adv2(ratio):
    '''
    random choose some validation pairs from sim tasks and pertube their human scores
    to check the robustness of the hyperparameters of the AR and HRSWE models
    :param ratio: num of validation pairs chosen / num of all validation pairs
    :return:
    '''
    dev_pairs = []

    with open('task_data/SimVerb-500-dev.txt', 'r') as f:
        for line in f:
            dev_pairs.append(line.strip().split('\t'))

    # random select a subset of pairs and change their scores
    sel_idx = random.sample(range(len(dev_pairs)), int(len(dev_pairs) * ratio))
    for id in sel_idx:
        dev_pairs[id][3] = str(random.randint(0, 1000) / 100)

    # save the adv pairs
    with open('task_data/SimVerb-500-dev-adv.txt', 'w') as f:
        for pair in dev_pairs:
            f.write('\t'.join(pair) + '\n')


def generate_adv3(ratio, thesauri, tasks):
    '''
    1. extract task pairs
    2. intersect the thesauri pairs with task pairs
    3. put some intersected synonym pairs into antonym constrain pairs
    this adversarial method could be more accurate than the adv1
    :param ratio: number of positive(syn) tuples chosen from intersected synonym pairs / num of all intersected synonyms
    :param thesauri: dict, syn_fname/ant_fname : names of files that contain constrains used to specialize the embedding
    :param tasks: dict, tasks used to extract the pairs
    :return:
    '''
    task_tuples = set()
    reverse_task_tuples = set()
    for k,v in tasks.items():
       task_tuples |= set(tuple(x) for x in v['X'])
       reverse_task_tuples |= set(tuple(x[::-1]) for x in v['X'])

    syn_pairs, ant_pairs = set(), set()
    with open(thesauri['syn_fname'], 'r') as f_syn:
        for line in f_syn:
            word_pair = line.split()
            word_pair = tuple(word[3:] for word in word_pair)  # remove the 'en-' prefix
            syn_pairs.add(word_pair)
    with open(thesauri['ant_fname'], 'r') as f_ant:
        for line in f_ant:
            word_pair = line.split()
            word_pair = tuple(word[3:] for word in word_pair)  # remove the 'en-' prefix
            ant_pairs.add(word_pair)

    inter_syn_pairs = task_tuples & syn_pairs
    inter_syn_pairs |= reverse_task_tuples & syn_pairs
    inter_ant_pairs = task_tuples & ant_pairs
    inter_ant_pairs |= reverse_task_tuples & ant_pairs

    # adversarial approach 1
    # choose a portion of sel_syn_pairs and put them into antonym pairs
    subset = set(random.sample(inter_syn_pairs,int(len(inter_syn_pairs)*ratio)))
    syn_pairs -= subset
    ant_pairs |= subset

    # write into files
    pa,fname = path.split(thesauri['syn_fname'])
    with open(join(pa,'adv_'+fname), 'w') as f_syn_out:
        for line in syn_pairs:
            f_syn_out.write(' '.join('en_'+w for w in line)+'\n')
    pa, fname = path.split(thesauri['ant_fname'])
    with open(join(pa,'adv_'+fname), 'w') as f_ant_out:
        for line in ant_pairs:
            f_ant_out.write(' '.join('en_'+w for w in line)+'\n')


def load_SimVerb3500(simverb_fname):
    X, y = [], []
    with open(simverb_fname, 'r') as f:
        for line in f:
            try:
                w1, w2, _, score, _ = line.strip().split('\t')
            except ValueError:
                w1, w2, _, score = line.strip().split('\t')
            X.append([w1, w2])
            y.append(score)
    X = np.vstack(X)
    y = np.array(y, dtype=float)
    return Bunch(X=X.astype("object"), y=y)

def load_syn_ant_classify(classification_fname):
    X, y = [], []
    with open(classification_fname, 'r') as f:
        for l in f:
            # 0: synonym 1: antonym
            w1,w2,label = l.strip().split('\t')
            X.append([w1,w2])
            y.append(label)
    X = np.vstack(X)
    y = np.array(y, dtype=int)
    return Bunch(X=X.astype("object"), y=y)