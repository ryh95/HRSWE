import itertools
import random
import time
from collections import Counter

import networkx as nx
import numpy as np


from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

from utils import nearestPD
import tensorflow as tf


def generate_syn_ant_graph(words, syn_pairs, ant_pairs):
    '''
    add words as nodes, use thesauri to add 1/-1 edges to nodes
    :return: adj graph of the positive and negative graph
    '''
    G = nx.Graph()
    G.add_nodes_from(words)

    positive_edges = ((k, v, 1) for k, v in syn_pairs)
    negative_edges = ((k, v, -1) for k, v in ant_pairs)
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
    adj_neg[adj < 0] = 1
    adj_neg.eliminate_zeros()
    return adj_pos, adj_neg, G

'''
separate syn ant graph version of generate_syn_ant_graph
'''
# def generate_syn_ant_graph(self, words, syn_pairs, ant_pairs):
#     '''
#     add words as nodes, use thesauri to add 1/-1 edges to nodes
#     :return: adj graph of the positive and negative graph
#     '''
#     G_syn = nx.Graph()
#     G_syn.add_nodes_from(words)
#     positive_edges = ((k, v, 1) for k, v in syn_pairs)
#     G_syn.add_weighted_edges_from(positive_edges)
#     # If a pair of words has positive edge and negative edge, the positive edge will be removed
#     # then the previous post-processing step can be removed
#     G_ant = nx.Graph()
#     G_ant.add_nodes_from(words)
#     negative_edges = ((k, v, 1) for k, v in ant_pairs)
#     G_ant.add_weighted_edges_from(negative_edges)
#
#     adj_pos = nx.adjacency_matrix(G_syn, nodelist=words)
#     adj_neg = nx.adjacency_matrix(G_ant, nodelist=words)
#
#     return adj_pos, adj_neg

def generate_spread_graph_1(G_thes):

    # only consider the three nodes types
    G_spread = nx.Graph()
    G_spread.nodes()
    G_spread.add_nodes_from(G_thes.nodes)
    i_k_syn_counter = Counter()
    i_k_ant_counter = Counter()

    for w_i in G_thes.nodes:
        for w_j in G_thes[w_i]:
            req_w_k = (w_k for w_k in G_thes[w_j] if w_k not in G_thes[w_i] and w_k not in (w_i, w_j))
            for w_k in req_w_k:

                if G_thes[w_i][w_j]['weight'] * G_thes[w_j][w_k]['weight'] > 0:

                    G_spread.add_edge(w_i, w_k, weight=1) # add an edge

                    i_k_syn_counter.update([frozenset((w_i, w_k))])

                else:

                    if w_k in G_spread[w_i]:
                        # if i,k is linked in spread graph
                        # remove this link
                        G_spread.remove_edge(w_i,w_k)

                    i_k_ant_counter.update([frozenset((w_i, w_k))])

    return G_spread

def generate_spread_graph_2(G_thes):

    # only consider the three nodes types
    G_spread = nx.Graph()
    G_spread.add_nodes_from(G_thes.nodes)
    i_k_syn_counter = Counter()
    i_k_ant_counter = Counter()

    for w_i in G_thes.nodes:
        for w_j in G_thes[w_i]:
            req_w_k = (w_k for w_k in G_thes[w_j] if w_k not in G_thes[w_i] and w_k not in (w_i, w_j))
            for w_k in req_w_k:

                if G_thes[w_i][w_j]['weight'] == 1 and G_thes[w_j][w_k]['weight'] == 1:

                    G_spread.add_edge(w_i, w_k, weight=1)  # add an edge

                    i_k_syn_counter.update([frozenset((w_i, w_k))])

                elif G_thes[w_i][w_j]['weight'] * G_thes[w_j][w_k]['weight'] < 0:

                    if w_k in G_spread[w_i]:
                        # if i,k is linked in spread graph
                        # remove this link
                        G_spread.remove_edge(w_i, w_k)

                    i_k_ant_counter.update([frozenset((w_i, w_k))])

    return G_spread

def generate_spread_graph_3(G_thes):

    G_spread = nx.Graph()
    G_spread.add_nodes_from(G_thes.nodes)

    for w in G_thes.nodes:
        # fully connect a node's neighbor
        # ref: https://stackoverflow.com/a/10651524/6609622
        # todo: fix this later, since not all neighbor edges are syn
        G_spread.add_edges_from(itertools.combinations(G_thes[w], 2))

    for e in G_spread.edges:
        # todo: solve dict change issue
        # if the added edge is a negative edge
        # withdraw the adding
        if G_thes.get_edge_data(*e,default={}).get('weight') == -1:
            G_spread.remove_edge(*e)

    return G_spread

def generate_spread_graph_4(G_thes,dataset):

    # emb = [vec for vec in dataset.emb_dict.values()]
    # emb = np.vstack(emb).astype(np.float32).T
    #
    # # normalize vector
    # emb_norm = np.linalg.norm(emb, axis=0)[np.newaxis, :]
    # emb = emb / emb_norm
    # # print(np.linalg.norm(emb,axis=0)[np.newaxis,:])
    #
    # W = emb.T @ emb
    # W = np.abs(W)


    adj = nx.adjacency_matrix(G_thes, nodelist=dataset.words)
    for i in range(1):
        # adj = adj.multiply(W)
        adj = adj @ adj
        adj = np.tanh(adj)
        # adj = adj.multiply((adj <= 1).multiply(adj >= -1))

    G_spread = nx.from_numpy_array(adj.toarray())
    id2words = {id:word for id,word in enumerate(dataset.words)}
    G_spread = nx.relabel_nodes(G_spread,id2words)

    return G_spread

def generate_spread_graph_5(G_thes):

    '''
    a random walk based generation
    :param G_thes:
    :return:
    '''

    G_spread = nx.Graph()
    G_spread.add_nodes_from(G_thes.nodes)

    nodes = G_thes.nodes
    nodes = list(nodes)
    # sel_nodes = random.sample(nodes,int(2*len(nodes)))

    for _ in tqdm(range(2*len(nodes))):
        cur_node = random.choice(nodes)
        # skip the no neighbor nodes
        if not list(G_thes.neighbors(cur_node)): continue

        # some hepler structure
        passed_nodes = set()
        syn_set = set()
        ant_set = set()
        # start from a random node
        passed_nodes.add(cur_node)
        for _ in range(5): # random walk 10 steps from the init node
            # if the current node has no neighbor, break the loop
            avail_nodes = set(G_thes.neighbors(cur_node)) - passed_nodes
            if not avail_nodes: break

            # random choose a neighbor from the node
            next_node = random.choice(list(avail_nodes))
            passed_nodes.add(next_node)

            if cur_node not in ant_set:
                if G_thes[cur_node][next_node]['weight'] == 1:
                    syn_set.add(next_node)
                else:
                    ant_set.add(next_node)
            else:
                if G_thes[cur_node][next_node]['weight'] == 1:
                    ant_set.add(next_node)
                else:
                    syn_set.add(next_node)

            cur_node = next_node

        # add edges within syn and ant and between syn and ant
        G_spread.add_edges_from(itertools.combinations(syn_set, 2),weight=1)
        G_spread.add_edges_from(itertools.combinations(ant_set, 2),weight=1)
        G_spread.add_edges_from(itertools.product(syn_set,ant_set),weight=-1)

        # post processing the added edge
        for e in list(G_spread.edges(passed_nodes,data='weight')):
            # if the spread edge is conflict with the original thesauri, remove the spread one
            if G_thes.get_edge_data(*e[:2], default={}).get('weight',0) * e[2] < 0:
                G_spread.remove_edge(*e[:2])

    return G_spread

def generate_spread_graph(G_thes,mis_syn,mis_ant):

    # only consider the three nodes types
    G_spread = nx.Graph()
    G_spread.add_nodes_from(G_thes.nodes)
    i_k_syn_counter = Counter()
    i_k_ant_counter = Counter()

    for w_i in G_thes.nodes:
        for w_j in G_thes[w_i]:
            req_w_k = (w_k for w_k in G_thes[w_j] if w_k not in G_thes[w_i] and w_k not in (w_i, w_j))
            for w_k in req_w_k:

                if G_thes[w_i][w_j]['weight'] * G_thes[w_j][w_k]['weight'] > 0:

                    if w_k not in G_spread[w_i]:
                        G_spread.add_edge(w_i,w_k,weight=mis_syn)
                    else:
                        G_spread[w_i][w_k]['weight'] += mis_syn ** (i_k_syn_counter[frozenset((w_i,w_k))]+1)
                    i_k_syn_counter.update([frozenset((w_i,w_k))])

                else:

                    if w_k not in G_spread[w_i]:
                        G_spread.add_edge(w_i, w_k, weight=-mis_ant)
                    else:
                        G_spread[w_i][w_k]['weight'] -= mis_ant ** (i_k_ant_counter[frozenset((w_i,w_k))]+1)
                    i_k_ant_counter.update([frozenset((w_i, w_k))])


    return G_spread

class HRSWE(object):

    def __init__(self, *hyps, **kwargs):

        # self.beta0, self.beta1, self.beta2, self.mis_syn = hyps
        self.beta0, self.beta1, self.beta2, self.mis_syn, self.pos, self.neg = hyps
        # self.beta0, self.beta1, self.beta2 = hyps
        # beta0,beta1, beta2,beta3,beta4 = hyps

        # self.beta3 = beta3
        # self.beta4 = beta4

        self.adj_pos = kwargs['adj_pos']
        self.adj_neg = kwargs['adj_neg']
        self.adj_spread = kwargs['adj_spread']
        self.W = kwargs['W']
        self.n = kwargs['n']
        self.d = kwargs['d']

    def generate_spread_graph(self):
        '''
        generate graph 6
        :return:
        '''
        adj = self.adj_spread.astype(float)
        adj[adj>0] = self.pos
        adj[adj<0] = -self.neg
        for i in range(1):
            # adj = adj.multiply(self.W)
            adj = adj @ adj
            # adj = np.tanh(adj)
            # adj = adj.multiply((adj <= 1).multiply(adj >= -1))
            adj[adj>1] = 1
            adj[adj<-1] = -1

        adj.eliminate_zeros()
        return adj

    def specialize_emb(self,emb_dict,syn_pairs,ant_pairs):
        """
        Dynamic Adjusted Word Embedding

        :param float beta:
        :param list words:
        :param nxd ndarray emb:
        :param dict thesauri: keys, [name, word_sim_pairs(if you use word sim data to create graph)]
        :param dict config:  keys, [sim_mat_type(pd/n, n means normal), eig_vec_option(ld, largest eig value),
        emb_type]

        :return: nxd ndarray new word embedding
        """
        W,n,d = self.W,self.n,self.d
        adj_spread = self.generate_spread_graph()

        # adj_spread[self.adj_pos>0] = 0
        # adj_spread[self.adj_neg>0] = 0

        adj_pos_spread = adj_spread.copy()
        adj_pos_spread[adj_spread < 0] = 0
        adj_pos_spread.eliminate_zeros()
        # id_changed = set(zip(*adj_pos_spread.nonzero())) & set(zip(*self.adj_pos.nonzero()))
        # if id_changed:
        #     adj_pos_spread[tuple(zip(*id_changed))] = 0
        #     adj_pos_spread.eliminate_zeros()

        # data = np.ones(adj_pos_spread.nnz)
        # adj_pos_spread_id = csr_matrix((data,adj_pos_spread.nonzero()),adj_pos_spread.shape)
        # sel_id = adj_pos_spread_id - self.adj_pos
        # adj_pos_spread = adj_pos_spread.multiply(sel_id)

        adj_neg_spread = adj_spread.copy()
        adj_neg_spread[adj_spread > 0] = 0
        adj_neg_spread.eliminate_zeros()
        # id_changed = set(zip(*adj_neg_spread.nonzero())) & set(zip(*self.adj_neg.nonzero()))
        # if id_changed:
        #     adj_neg_spread[tuple(zip(*id_changed))] = 0
        #     adj_neg_spread.eliminate_zeros()

        # adj_neg_spread[adj_spread < 0] = 1


        # adj_spread_pos = adj_spread[adj_spread>0]
        # adj_spread_neg = adj_spread[adj_spread<0]

        # W_prime = self.beta0*W + self.beta1 * self.adj_pos.multiply(1 - W) - self.beta2 * self.adj_neg.multiply(W - (-1))
        # W_prime = self.beta0*W + self.beta1 * self.adj_pos.multiply(1 - W) - self.beta2 * self.adj_neg.multiply(W - (-1)) + self.mis_syn * self.adj_spread

        # W_prime = self.beta0 * W + self.beta1 * self.adj_pos.multiply(1 - W) - self.beta1 * self.adj_neg.multiply(
        #     W - (-1)) + self.mis_syn * adj

        # W_prime = self.beta0 * W + self.beta1 * self.adj_pos.multiply(np.max(W) - W) - self.beta1 * self.adj_neg.multiply(W - np.min(W)) + \
        #           self.mis_syn * adj_spread.multiply(W)

        W_prime = self.beta0 * W + self.beta1 * self.adj_pos.multiply(
            np.max(W) - W) - self.beta2 * self.adj_neg.multiply(W - np.min(W)) + \
                 self.beta1 * adj_pos_spread.multiply(np.max(W) - W) + self.beta2 * adj_neg_spread.multiply(W - np.min(W))

        # W_prime = self.beta0 * W + self.beta1 * self.adj_pos.multiply(W) - self.beta2 * self.adj_neg.multiply(W) # rswe-2

        # W_prime = self.beta0 * W + self.beta1 * self.adj_pos.multiply(W) - self.beta2 * self.adj_neg.multiply(W) + \
        #           self.beta1 * adj_pos_spread.multiply(W) + self.beta2 * adj_neg_spread.multiply(W)  # rswe-3

        # W_prime = self.beta0 * W + self.beta1 * self.adj_pos.multiply(np.max(W) - W) - self.beta2 * self.adj_neg.multiply(W - np.min(W))

        # W_prime = self.beta0 * W + self.beta1 * self.pos * self.adj_pos * (
        #     np.max(W) - W) - self.beta1 * np.abs(self.neg) * self.adj_neg * (W - np.min(W)) + \
        #           self.beta1 * adj_pos_spread.multiply(np.max(W) - W) + self.beta1 * adj_neg_spread.multiply(
        #     W - np.min(W))

        # W_prime = self.beta0*W + self.beta1 * self.adj_pos.multiply(np.max(W) - W) - self.beta2 * self.adj_neg.multiply(W - np.min(W)) + \
        #           self.mis_syn * self.adj_spread.multiply(W)

        # W_prime = self.beta0 * W + self.beta1 * adj_pos.multiply(1 - W) - self.beta2 * adj_neg.multiply(
        #     W - (-1))
        # W_prime = self.beta0*W + self.beta1 * adj_pos.multiply(np.max(W) - self.beta3*W) + self.beta2 * adj_neg.multiply(W - self.beta4*np.min(W))
        # W_prime = self.beta0 * W - self.beta1 * self.adj_pos.multiply(W) - self.beta2 * self.adj_neg.multiply(W) + \
        #           self.beta1 * self.adj_pos.multiply(1) + self.beta2 * self.adj_neg.multiply(-1) + self.adj_spread

        # W_prime = np.clip(W_prime, -1, 1)

        scaler = StandardScaler()
        W_prime = scaler.fit_transform(W_prime)
        # W_prime = W_prime - W_prime.mean(axis=0)

        # W_hat = nearestPD(W_prime)
        W_prime = (W_prime + W_prime.T)/2

        # choose d largest eigenvectors
        # ref: https://stackoverflow.com/a/12168664
        # turbo: use divide and conquer algorithm or not
        lamb_s, Q_s = linalg.eigh(W_prime, eigvals=(n - d, n - 1))

        Q_s = Q_s[:,::-1]
        lamb_s = lamb_s[::-1]
        lamb_s[lamb_s < 0] = 0

        new_emb = Q_s * (lamb_s ** (1 / 2))

        # U, Sigma, VT = randomized_svd(W_prime,n_components=d,n_iter=100) # svd method 1

        # U, Sigma, VT = np.linalg.svd(W_prime, full_matrices=False, hermitian=True) # svd method 2
        # U = U[:,:d]
        # Sigma = Sigma[:d]
        # VT = VT[:d,:]

        # new_emb = (np.diag(Sigma) @ VT).T
        # new_emb = U @ np.diag(Sigma)
        print(new_emb.shape)

        return new_emb

class RetrofittedMatrix(object):

    def __init__(self, *hyps,**kwargs):
        # self.beta0, self.beta1, self.beta2, self.mis_syn = hyps
        self.beta0, self.beta1, self.beta2, self.mis_syn, self.pos, self.neg = hyps
        # self.beta0, self.beta1, self.beta2, self.mis_syn, self.ths = hyps

        self.adj_pos = kwargs['adj_pos']
        self.adj_neg = kwargs['adj_neg']
        self.adj_spread = kwargs['adj_spread']
        self.W = kwargs['W']

    def generate_spread_graph(self):
        '''
        generate graph 6
        :return:
        '''
        adj = self.adj_spread.astype(float)
        adj[adj>0] = self.pos
        adj[adj<0] = self.neg
        for i in range(1):
            # adj = adj.multiply(W)
            adj = adj @ adj
            adj = np.tanh(adj)
            # adj = adj.multiply((adj <= 1).multiply(adj >= -1))

        return adj
    def specialize_emb(self,emb_dict,syn_pairs,ant_pairs):
        W = self.W
        adj = self.generate_spread_graph()
        # W_ths = np.zeros_like(W)
        # W_ths[np.abs(W)>=self.ths] = 1

        # W_prime = self.beta0 * W + self.beta1 * self.adj_pos.multiply(1 - W) - self.beta2 * self.adj_neg.multiply(
        #     W - (-1))
        # W_prime = self.beta0 * W + self.beta1 * self.adj_pos.multiply(1 - W) - self.beta1 * self.adj_neg.multiply(
        #     W - (-1)) + self.mis_syn * adj

        W_prime = self.beta0 * W + self.beta1 * (self.adj_pos @ self.adj_pos) - self.beta2 * (self.adj_neg @ self.adj_neg) + self.mis_syn * adj

        # W_prime = self.beta0 * W + self.beta1 * self.adj_pos.multiply(
        #     np.max(W) - W) - self.beta2 * self.adj_neg.multiply(W - np.min(W)) + \
        #           self.mis_syn * self.adj_spread.multiply(W)

        W_prime = np.clip(W_prime,-1,1)

        # scaler = StandardScaler()
        # W_prime = scaler.fit_transform(W_prime)

        return W_prime

class LHRSWE(HRSWE):

    def specialize_emb(self,emb_dict,syn_pairs,ant_pairs):

        words = [w for w in emb_dict.keys()]
        emb = [vec for vec in emb_dict.values()]
        emb = np.vstack(emb).astype(np.float32).T

        d,n = emb.shape

        # normalize vector
        emb_norm = np.linalg.norm(emb,axis=0)[np.newaxis,:]
        emb = emb / emb_norm
        # print(np.linalg.norm(emb,axis=0)[np.newaxis,:])

        W = emb.T @ emb

        adj_pos, adj_neg = self.generate_syn_ant_graph(words, syn_pairs, ant_pairs)

        W_prime = self.beta0 * W + self.beta1 * adj_pos.multiply(1 - W) - self.beta2 * adj_neg.multiply(
            W - (-1))

        W_prime = np.clip(W_prime,-1,1)

        lamb_s, Q_s = linalg.eigh(W_prime, eigvals=(n - d, n - 1))
        # todo: normalize Q_s if needed
        # row_norm = np.linalg.norm(Q_s, axis=1)[:, np.newaxis]
        # Q_s = Q_s / row_norm
        return Q_s

class AR(object):

    def __init__(self, *hyps):
        """
        To initialise the class, we need to supply the config file, which contains the location of
        the pretrained (distributional) word vectors, the location of (potentially more than one)
        collections of linguistic constraints (one pair per line), as well as the
        hyperparameters of the Attract-Repel procedure (as detailed in the TACL paper).
        """
        syn_mar, ant_mar, batch_size, epoch_num, l2_reg = hyps
        self.attract_margin_value = syn_mar
        self.repel_margin_value = ant_mar
        self.regularisation_constant_value = l2_reg
        self.batch_size = batch_size
        self.max_iter = epoch_num

        print("\nExperiment hyperparameters (attract_margin, repel_margin, batch_size, l2_reg_constant, max_iter):", \
              self.attract_margin_value, self.repel_margin_value, self.batch_size, self.regularisation_constant_value,
              self.max_iter)

    def initialise_model(self, numpy_embedding):
        """
        Initialises the TensorFlow Attract-Repel model.
        """
        self.attract_examples = tf.placeholder(tf.int32, [None, 2])  # each element is the position of word vector.
        self.repel_examples = tf.placeholder(tf.int32, [None, 2])  # each element is again the position of word vector.

        self.negative_examples_attract = tf.placeholder(tf.int32, [None, 2])
        self.negative_examples_repel = tf.placeholder(tf.int32, [None, 2])

        self.attract_margin = tf.placeholder("float")
        self.repel_margin = tf.placeholder("float")
        self.regularisation_constant = tf.placeholder("float")

        # Initial (distributional) vectors. Needed for L2 regularisation.
        self.W_init = tf.constant(numpy_embedding, name="W_init") # nxd

        # Variable storing the updated word vectors.
        self.W_dynamic = tf.Variable(numpy_embedding, name="W_dynamic") # nxd

        # Attract Cost Function:

        # placeholders for example pairs...
        # IMPORTANT: the embeddings are all normalized
        attract_examples_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.attract_examples[:, 0]),
                                                   1)
        attract_examples_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.attract_examples[:, 1]),
                                                    1)

        # and their respective negative examples:
        negative_examples_attract_left = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_attract[:, 0]), 1)
        negative_examples_attract_right = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_attract[:, 1]), 1)

        # dot product between the example pairs.
        attract_similarity_between_examples = tf.reduce_sum(tf.multiply(attract_examples_left, attract_examples_right),
                                                            1)

        # dot product of each word in the example with its negative example.
        attract_similarity_to_negatives_left = tf.reduce_sum(
            tf.multiply(attract_examples_left, negative_examples_attract_left), 1)
        attract_similarity_to_negatives_right = tf.reduce_sum(
            tf.multiply(attract_examples_right, negative_examples_attract_right), 1)

        # and the final Attract Cost Function (sans regularisation):
        self.attract_cost = tf.nn.relu(
            self.attract_margin + attract_similarity_to_negatives_left - attract_similarity_between_examples) + \
                            tf.nn.relu(
                                self.attract_margin + attract_similarity_to_negatives_right - attract_similarity_between_examples)

        # Repel Cost Function:

        # placeholders for example pairs...
        repel_examples_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.repel_examples[:, 0]),
                                                 1)  # becomes batch_size X vector_dimension
        repel_examples_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.repel_examples[:, 1]), 1)

        # and their respective negative examples:
        negative_examples_repel_left = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_repel[:, 0]), 1)
        negative_examples_repel_right = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_repel[:, 1]), 1)

        # dot product between the example pairs.
        repel_similarity_between_examples = tf.reduce_sum(tf.multiply(repel_examples_left, repel_examples_right),
                                                          1)  # becomes batch_size again, might need tf.squeeze

        # dot product of each word in the example with its negative example.
        repel_similarity_to_negatives_left = tf.reduce_sum(
            tf.multiply(repel_examples_left, negative_examples_repel_left), 1)
        repel_similarity_to_negatives_right = tf.reduce_sum(
            tf.multiply(repel_examples_right, negative_examples_repel_right), 1)

        # and the final Repel Cost Function (sans regularisation):
        self.repel_cost = tf.nn.relu(
            self.repel_margin - repel_similarity_to_negatives_left + repel_similarity_between_examples) + \
                          tf.nn.relu(
                              self.repel_margin - repel_similarity_to_negatives_right + repel_similarity_between_examples)

        # The Regularisation Cost (separate for the two terms, depending on which one is called):

        # load the original distributional vectors for the example pairs:
        original_attract_examples_left = tf.nn.embedding_lookup(self.W_init, self.attract_examples[:, 0])
        original_attract_examples_right = tf.nn.embedding_lookup(self.W_init, self.attract_examples[:, 1])

        original_repel_examples_left = tf.nn.embedding_lookup(self.W_init, self.repel_examples[:, 0])
        original_repel_examples_right = tf.nn.embedding_lookup(self.W_init, self.repel_examples[:, 1])

        # and then define the respective regularisation costs:
        regularisation_cost_attract = self.regularisation_constant * (
                    tf.nn.l2_loss(original_attract_examples_left - attract_examples_left) + tf.nn.l2_loss(
                original_attract_examples_right - attract_examples_right))
        self.attract_cost += regularisation_cost_attract

        regularisation_cost_repel = self.regularisation_constant * (
                    tf.nn.l2_loss(original_repel_examples_left - repel_examples_left) + tf.nn.l2_loss(
                original_repel_examples_right - repel_examples_right))
        self.repel_cost += regularisation_cost_repel

        # Finally, we define the training step functions for both steps.

        tvars = tf.trainable_variables()
        attract_grads = [tf.clip_by_value(grad, -2., 2.) for grad in tf.gradients(self.attract_cost, tvars)]
        # debug: check if gradients are only calculated for the words in lexical constraints
        # self.un_clipped_attract_grads = tf.gradients(self.attract_cost, tvars)
        repel_grads = [tf.clip_by_value(grad, -2., 2.) for grad in tf.gradients(self.repel_cost, tvars)]

        attract_optimiser = tf.train.AdagradOptimizer(0.05)
        repel_optimiser = tf.train.AdagradOptimizer(0.05)

        self.attract_cost_step = attract_optimiser.apply_gradients(list(zip(attract_grads, tvars)))
        self.repel_cost_step = repel_optimiser.apply_gradients(list(zip(repel_grads, tvars)))

        # return the handles for loading vectors from the TensorFlow embeddings:
        return attract_examples_left, attract_examples_right, repel_examples_left, repel_examples_right

    def extract_negative_examples(self, list_minibatch, attract_batch=True):
        """
        For each example in the minibatch, this method returns the closest vector which is not
        in each words example pair.
        """

        list_of_representations = []
        list_of_indices = []

        representations = self.sess.run([self.embedding_attract_left, self.embedding_attract_right],
                                        feed_dict={self.attract_examples: list_minibatch})

        for idx, (example_left, example_right) in enumerate(list_minibatch):
            list_of_representations.append(representations[0][idx])
            list_of_representations.append(representations[1][idx])

            list_of_indices.append(example_left)
            list_of_indices.append(example_right)

        condensed_distance_list = pdist(list_of_representations, 'cosine')
        square_distance_list = squareform(condensed_distance_list)

        if attract_batch:
            default_value = 2.0  # value to set for given attract/repel pair, so that it can not be found as closest or furthest away.
        else:
            default_value = 0.0  # for antonyms, we want the opposite value from the synonym one. Cosine Distance is [0,2].

        for i in range(len(square_distance_list)):

            square_distance_list[i, i] = default_value

            if i % 2 == 0:
                square_distance_list[i, i + 1] = default_value
            else:
                square_distance_list[i, i - 1] = default_value

        if attract_batch:
            negative_example_indices = np.argmin(square_distance_list,
                                                    axis=1)  # for each of the 100 elements, finds the index which has the minimal cosine distance (i.e. most similar).
        else:
            negative_example_indices = np.argmax(square_distance_list,
                                                    axis=1)  # for antonyms, find the least similar one.

        negative_examples = []

        for idx in range(len(list_minibatch)):
            negative_example_left = list_of_indices[negative_example_indices[2 * idx]]
            negative_example_right = list_of_indices[negative_example_indices[2 * idx + 1]]

            negative_examples.append((negative_example_left, negative_example_right))

        # todo: try to remove mix_sampling
        negative_examples = self.mix_sampling(list_minibatch, negative_examples)

        return negative_examples

    def mix_sampling(self,list_of_examples, negative_examples):
        """
        Converts half of the negative examples to random words from the batch (that are not in the given example pair).
        """
        mixed_negative_examples = []
        batch_size = len(list_of_examples)

        for idx, (left_idx, right_idx) in enumerate(negative_examples):

            new_left = left_idx
            new_right = right_idx

            if random.random() >= 0.5:
                new_left = list_of_examples[self.random_different_from(batch_size, idx)][random.randint(0, 1)]

            if random.random() >= 0.5:
                new_right = list_of_examples[self.random_different_from(batch_size, idx)][random.randint(0, 1)]

            mixed_negative_examples.append((new_left, new_right))

        return mixed_negative_examples

    def random_different_from(self,top_range, number_to_not_repeat):
        result = random.randint(0, top_range - 1)
        while result == number_to_not_repeat:
            result = random.randint(0, top_range - 1)

        return result

    def obtain_sp_vec(self):
        """
        Extracts the current word vectors from TensorFlow embeddings and (if print_simlex=True) prints their SimLex scores.
        """

        return self.sess.run(self.W_dynamic)

    def specialize_emb(self,emb_dict,syn_pairs,ant_pairs):

        """
        This method repeatedly applies optimisation steps to fit the word vectors to the provided linguistic constraints.
        """
        emb = [vec for vec in emb_dict.values()]
        emb = np.vstack(emb).astype(np.float32)

        word2id = {w: i for i, w in enumerate(emb_dict.keys())}
        self.syn_pairs = [(word2id[w1],word2id[w2]) for w1,w2 in syn_pairs]
        self.ant_pairs = [(word2id[w1], word2id[w2]) for w1, w2 in ant_pairs]

        # load the handles so that we can load current state of vectors from the Tensorflow embedding.
        tf.reset_default_graph()
        embedding_handles = self.initialise_model(emb)

        self.embedding_attract_left = embedding_handles[0]
        self.embedding_attract_right = embedding_handles[1]
        self.embedding_repel_left = embedding_handles[2]
        self.embedding_repel_right = embedding_handles[3]

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)

        # specialize embedding
        current_iteration = 0


        # TODO: if a pair (w_a,w_b) is in synonyms, (w_b,w_a) can also in synonyms.
        # figure out how many pairs have this problem

        syn_count = len(self.syn_pairs)
        ant_count = len(self.ant_pairs)

        print("\nAntonym pairs:", len(self.ant_pairs), "Synonym pairs:", len(self.syn_pairs))

        syn_batches = int(syn_count / self.batch_size)
        ant_batches = int(ant_count / self.batch_size)

        batches_per_epoch = syn_batches + ant_batches

        print("\nRunning the optimisation procedure for", self.max_iter, "iterations...")
        start = time.time()
        last_time = time.time()


        while current_iteration < self.max_iter:

            # how many attract/repel batches we've done in this epoch so far.
            antonym_batch_counter = 0
            synonym_batch_counter = 0

            order_of_synonyms = list(range(0, syn_count))
            order_of_antonyms = list(range(0, ant_count))
            # IMPORTANT: before training the syn/ant constraints will be shuffled
            random.shuffle(order_of_synonyms)
            random.shuffle(order_of_antonyms)

            # list of 0 where we run synonym batch, 1 where we run antonym batch
            # IMPORTANT: training steps of syn and ant mini batches will be shuffled
            list_of_batch_types = [0] * batches_per_epoch
            list_of_batch_types[syn_batches:] = [1] * ant_batches  # all antonym batches to 1
            random.shuffle(list_of_batch_types)

            if current_iteration == 0:
                print("\nStarting epoch:", current_iteration + 1, "\n")
            else:
                print("\nStarting epoch:", current_iteration + 1, "Last epoch took:",
                      round(time.time() - last_time, 1),
                      "seconds. \n")
                last_time = time.time()

            for batch_index in range(0, batches_per_epoch):

                syn_or_ant_batch = list_of_batch_types[batch_index]

                if syn_or_ant_batch == 0:
                    # do one synonymy batch:

                    synonymy_examples = [self.syn_pairs[order_of_synonyms[x]] for x in
                                         range(synonym_batch_counter * self.batch_size,
                                               (synonym_batch_counter + 1) * self.batch_size)]
                    current_negatives = self.extract_negative_examples(synonymy_examples, attract_batch=True)

                    self.sess.run([self.attract_cost_step], feed_dict={self.attract_examples: synonymy_examples,
                                                                       self.negative_examples_attract: current_negatives, \
                                                                       self.attract_margin: self.attract_margin_value,
                                                                       self.regularisation_constant: self.regularisation_constant_value})
                    synonym_batch_counter += 1

                else:

                    antonymy_examples = [self.ant_pairs[order_of_antonyms[x]] for x in
                                         range(antonym_batch_counter * self.batch_size,
                                               (antonym_batch_counter + 1) * self.batch_size)]
                    current_negatives = self.extract_negative_examples(antonymy_examples, attract_batch=False)

                    self.sess.run([self.repel_cost_step], feed_dict={self.repel_examples: antonymy_examples,
                                                                     self.negative_examples_repel: current_negatives, \
                                                                     self.repel_margin: self.repel_margin_value,
                                                                     self.regularisation_constant: self.regularisation_constant_value})

                    antonym_batch_counter += 1

            current_iteration += 1

        print("Training took ", round(time.time() - start, 1), "seconds.")

        return self.obtain_sp_vec()