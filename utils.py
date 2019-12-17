from collections import defaultdict
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

def get_all_words_in_constraint(word_sim_fname):
    vocab = set()
    with open(word_sim_fname, 'r') as f:
        for line in f:
            word_pair = line.split()
            word_pair = [word[3:] for word in word_pair]  # remove the 'en-' prefix
            vocab.update(word_pair)
    return vocab