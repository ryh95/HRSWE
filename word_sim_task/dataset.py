import pickle
import random
from os.path import join

from six import iteritems
from sklearn.utils import Bunch
from web.datasets.similarity import fetch_SimLex999

from constants import WORD_SIM_TASK_DIR, VOCAB_DIR, ORIGINAL_VECS_DIR, ORIGINAL_EMBEDDING
from preprocess import GeneralTextProcesser
import numpy as np
import os
import networkx as nx
import sys

class Dataset(object):

    def __init__(self,thesauri):

        self.thesauri = thesauri

    def load_SimVerb3500(self, simverb_fname):
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

    def generate_adv_thesaurus(self, syn_fname, ant_fname, po_ratio=0.5, reverse=False):
        syn_pairs, ant_pairs = set(), set()
        with open(syn_fname, 'r') as f_syn:
            for line in f_syn:
                syn_pairs.add(line)
        with open(ant_fname, 'r') as f_ant:
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

        head, syn_name = os.path.split(syn_fname)
        syn_out_fname = join(head, 'adv_' + syn_name)
        head, ant_name = os.path.split(ant_fname)
        ant_out_fname = join(head, 'adv_' + ant_name)
        with open(syn_out_fname, 'w') as f_syn_out:
            for line in syn_pairs:
                f_syn_out.write(line)
        with open(ant_out_fname, 'w') as f_ant_out:
            for line in ant_pairs:
                f_ant_out.write(line)

    def generate_adv_val(self,ratio):

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

    def load_datasets(self):
        self.sim_tasks = {
            # "MEN": fetch_MEN(),
            # "WS353": fetch_WS353(),
            # "WS353S": fetch_WS353(which="similarity"),
            "SIMLEX999": fetch_SimLex999(),
            "SIMVERB3000-test": self.load_SimVerb3500(join(WORD_SIM_TASK_DIR, 'task_data', 'SimVerb-3000-test.txt')),
            "SIMVERB500-dev": self.load_SimVerb3500(join(WORD_SIM_TASK_DIR, 'task_data', 'SimVerb-500-dev.txt')),
            # "RW": fetch_RW()
            # "RG65": fetch_RG65(),
            # "MTurk": fetch_MTurk(),
            # "TR9856": fetch_TR9856(),
        }
        self.vocab_fname = '_'.join(sorted(self.sim_tasks.keys()))


    def load_words(self):
        if os.path.isfile(join(VOCAB_DIR, self.vocab_fname + '.npy')):
            words = np.load(join(VOCAB_DIR, self.vocab_fname + '.npy'))
        else:

            words = set()
            for name, data in iteritems(self.sim_tasks):
                words |= set(data.X.flatten())

            # words, word_emb = get_words_vecs_from_word_sim(sim_tasks, embedding_dict)
            # add sentiment words

            words = list(words)

            # if test_on_sentiment:
            #
            #     sentiment_processer = StanfordSentimentProcesser()
            #     sentiment_vocab = set()
            #     for fname in ['train.txt', 'dev.txt', 'test.txt']:
            #         sentiment_vocab |= sentiment_processer.extract_vocab_from_processed_file(
            #             join(SENTIMENT_DIR, 'processed_' + fname))
            #
            #     words = list(set(words) | sentiment_vocab)

            np.save(join(VOCAB_DIR, self.vocab_fname + '.npy'), words)
        self.words = words

    def load_embeddings(self):
        emb_fname = join(ORIGINAL_VECS_DIR, self.vocab_fname)
        # self.emb_fname = 'paper_results/wn_ro_pd_ld_0'

        if os.path.isfile(emb_fname + '.pickle'):
            with open(emb_fname + '.pickle', 'rb') as handle:
                emb_dict = pickle.load(handle)

        else:
            text_preprocesser = GeneralTextProcesser()
            emb_dict = text_preprocesser.vocab2vec(self.words, ORIGINAL_VECS_DIR, self.vocab_fname, ORIGINAL_EMBEDDING,
                                                   ['pickle'], 'word2vec', normalize=True, oov_handle='mean_emb_vec')

        self.emb_dict = emb_dict

    def generate_syn_ant_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.words)
        words_set = set(self.words)
        if 'syn_fname' not in self.thesauri or 'ant_fname' not in self.thesauri:
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
        with open(self.thesauri['syn_fname'], 'r') as f:
            for line in f:
                word_pair = line.split()
                word_pair = [word[3:] for word in word_pair]  # remove the 'en-' prefix
                if word_pair[0] in words_set and word_pair[1] in words_set and word_pair[0] != word_pair[1]:
                    syn_constraints |= {(word_pair[0], word_pair[1])}

        ant_constraints = set()
        with open(self.thesauri['ant_fname'], 'r') as f:
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

        positive_edges = ((k, v, 1) for k, v in syn_constraints)
        negative_edges = ((k, v, -1) for k, v in ant_constraints)
        G.add_weighted_edges_from(positive_edges)
        # If a pair of words has positive edge and negative edge, the positive edge will be removed
        # then the previous post-processing step can be removed
        G.add_weighted_edges_from(negative_edges)

        adj = nx.adjacency_matrix(G, nodelist=self.words)

        adj_pos = adj.copy()
        adj_pos[adj < 0] = 0
        adj_pos.eliminate_zeros()

        adj_neg = adj.copy()
        adj_neg[adj > 0] = 0
        adj_neg.eliminate_zeros()
        return adj_pos, adj_neg