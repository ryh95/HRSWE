import pickle
from os.path import join

from six import iteritems
from web.datasets.similarity import fetch_SimLex999

from constants import WORD_SIM_TASK_DIR, VOCAB_DIR, ORIGINAL_VECS_DIR, ORIGINAL_EMBEDDING, SYN_ANT_CLASSIFY_TASK_DIR
from preprocess import GeneralTextProcesser
import numpy as np
import os
import networkx as nx
import sys

from utils import load_SimVerb3500, load_syn_ant_classify


class Dataset(object):

    def load_task_datasets(self,*names):
        '''

        :param names: names of task files
        :return:
        '''
        tasks = {}
        classification_datasets = set([i + '-pairs.' + j for j in ['test', 'val'] for i in ['adjective', 'noun', 'verb']])
        flag = 0
        for name in names:
            if name == 'SIMLEX999':
                tasks[name] = fetch_SimLex999()
            elif name == 'SIMVERB3000-test':
                tasks[name] = load_SimVerb3500(join(WORD_SIM_TASK_DIR, 'task_data', 'SimVerb-3000-test.txt'))
            elif name == 'SIMVERB500-dev':
                tasks[name] = load_SimVerb3500(join(WORD_SIM_TASK_DIR, 'task_data', 'SimVerb-500-dev.txt'))
            elif name in classification_datasets:
                tasks[name] = load_syn_ant_classify(join(SYN_ANT_CLASSIFY_TASK_DIR,'task_data',name))
                flag = 1

        if flag == 0:
            self.vocab_fname = '_'.join(sorted(tasks.keys()))
        else:
            self.vocab_fname = 'syn_ant_classify_test_val'
        self.tasks = tasks


    def load_words(self):
        '''
        load vocab from vocab_fname
        if no file, load from tasks
        '''
        if os.path.isfile(join(VOCAB_DIR, self.vocab_fname + '.npy')):
            words = np.load(join(VOCAB_DIR, self.vocab_fname + '.npy'))
        else:

            words = set()
            for name, data in iteritems(self.tasks):
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
        '''
        load embeddings from vocab_fname
        if no file, load from word2vec with words
        '''
        emb_fname = join(ORIGINAL_VECS_DIR, self.vocab_fname)
        # self.emb_fname = 'paper_results/wn_ro_pd_ld_0'

        if os.path.isfile(emb_fname + '.npy'):
            emb = np.load(emb_fname + '.npy')
        else:
            text_preprocesser = GeneralTextProcesser()
            _,emb = text_preprocesser.vocab2vec(self.words, ORIGINAL_VECS_DIR, self.vocab_fname, ORIGINAL_EMBEDDING,
                                                   ['pickle'], 'word2vec', normalize=True, oov_handle='mean_emb_vec')
        # nxd
        self.emb = emb

    def load_thesauri(self, fthesauri):
        '''
        load subset thesauri
        should use generate_sub_thesauri in utils to generate the subset first
        :param fthesauri: dict, syn_fname/ant_fname : names of files that contain constrains used to specialize the embedding
        :return:
        '''
        if 'syn_fname' not in fthesauri or 'ant_fname' not in fthesauri:
            # ref: https://stackoverflow.com/a/2950027
            sys.exit('No syn/ant found!')
        words_set = set(self.words)
        # if thesauri['name'] == 'word_sim':
        #     data = thesauri['word_sim_pairs']
        #     for (w1, w2), y in zip(data.X, data.y):
        #         if y > 8:
        #             G.add_edge(w1, w2, weight=1)
        #         elif y < 2:
        #             G.add_edge(w1, w2, weight=-1)
        self.syn_constraints = set()
        with open(fthesauri['syn_fname'], 'r') as f:
            for line in f:
                word_pair = line.split()
                word_pair = [word[3:] for word in word_pair]  # remove the 'en-' prefix
                if word_pair[0] in words_set and word_pair[1] in words_set and word_pair[0] != word_pair[1]:
                    self.syn_constraints |= {(word_pair[0], word_pair[1])}

        self.ant_constraints = set()
        with open(fthesauri['ant_fname'], 'r') as f:
            for line in f:
                word_pair = line.split()
                word_pair = [word[3:] for word in word_pair]  # remove the 'en-' prefix
                if word_pair[0] in words_set and word_pair[1] in words_set and word_pair[0] != word_pair[1]:
                    self.ant_constraints |= {(word_pair[0], word_pair[1])}

        # Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
        # to be consistent with attract_repel's processing
        # todo: tune the following
        for antonym_pair in self.ant_constraints:
            if antonym_pair in self.syn_constraints:
                self.syn_constraints.remove(antonym_pair)

    def generate_syn_ant_graph(self):
        '''
        add words as nodes, use thesauri to add 1/-1 edges to nodes
        :return: adj graph of the positive and negative graph
        '''
        G = nx.Graph()
        G.add_nodes_from(self.words)

        positive_edges = ((k, v, 1) for k, v in self.syn_constraints)
        negative_edges = ((k, v, -1) for k, v in self.ant_constraints)
        G.add_weighted_edges_from(positive_edges)
        # If a pair of words has positive edge and negative edge, the positive edge will be removed
        # then the previous post-processing step can be removed
        # todo: tune the following
        G.add_weighted_edges_from(negative_edges)

        adj = nx.adjacency_matrix(G, nodelist=self.words)

        adj_pos = adj.copy()
        adj_pos[adj < 0] = 0
        adj_pos.eliminate_zeros()

        adj_neg = adj.copy()
        adj_neg[adj > 0] = 0
        adj_neg.eliminate_zeros()
        return adj_pos, adj_neg