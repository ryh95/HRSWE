import numpy as np


from scipy import linalg

from utils import nearestPD


class HRSWE(object):

    def __init__(self,beta1, beta2, emb, adj_pos, adj_neg, config):

        self.beta1 = beta1
        self.beta2 = beta2
        self.emb = emb
        self.adj_pos = adj_pos
        self.adj_neg = adj_neg
        self.config = config

    def specialize_emb(self):
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

        d, n = self.emb.shape
        W = self.emb.T @ self.emb

        W_prime = W + self.beta1 * self.adj_pos.multiply(np.max(W) - W) + self.beta2 * self.adj_neg.multiply(W - np.min(W))

        if self.config['sim_mat_type'] == 'pd':
            W_hat = nearestPD(W_prime)
        elif self.config['sim_mat_type'] == 'n':
            W_hat = W_prime

        if self.config['eig_vec_option'] == 'ld':
            # choose d largest eigenvectors
            # ref: https://stackoverflow.com/a/12168664
            # turbo: use divide and conquer algorithm or not
            lamb_s, Q_s = linalg.eigh(W_hat, eigvals=(n - d, n - 1))

        if self.config['emb_type'] == 0:  # together use with pd
            new_emb = Q_s @ np.diag(lamb_s ** (1 / 2))
        elif self.config['emb_type'] == 1:  # together use with n
            row_norm = np.linalg.norm(Q_s, axis=1)[:, np.newaxis]
            new_emb = Q_s / row_norm
        # elif config['emb_type'] == 2: # together use with n
        #     assert np.all(lamb_s) >= 0
        #     new_emb = Q_s @ np.diag(lamb_s ** (1 / 2))
        return new_emb.T