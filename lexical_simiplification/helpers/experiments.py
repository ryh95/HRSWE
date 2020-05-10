import pickle
from datetime import datetime
from pathlib import Path

import networkx as nx
from sklearn.model_selection import train_test_split
from skopt import dump
from skopt.space import Integer, Real
import numpy as np
from experiments import BaseExperiments
from lexical_simiplification.helpers import text_embeddings
from lexical_simiplification.helpers import io_helper
from lexical_simiplification.helpers import lightls
from model import generate_syn_ant_graph


class LightLSExperiments(object):

    def __init__(self,config):

        self.config = config
        # preparation

        print("Loading unigram frequencies...")
        ls = io_helper.load_lines(config['fwordreqs'])
        self.wfs = {x.split()[0].strip(): int(x.split()[1].strip()) for x in ls}

        if config['fembs'].suffix == '.txt':
            self.embeddings = text_embeddings.Embeddings()
            self.embeddings.load_embeddings(config['fembs'], config['word_limit'], language='default', print_loading=True, skip_first_line=False,
                                         normalize=True)
            self.embeddings.inverse_vocabularies()
        elif config['fembs'].suffix == '.bin':
            self.embeddings = text_embeddings.Word2VecEmbedding()
            self.embeddings.load_embeddings(config['fembs'], limit=None, language='default', print_loading=True,
                                         skip_first_line=False, normalize=True)
        elif config['fembs'].suffix == '.pickle':
            self.embeddings = text_embeddings.PickleEmbedding()
            self.embeddings.load_embeddings(config['fembs'], config['word_limit'], language='default',
                                            print_loading=True, skip_first_line=False,
                                            normalize=True)
            # self.embeddings.inverse_vocabularies()


        self.stopwords = io_helper.load_lines(config['fstopwords']) if config['fstopwords'] else None

        with open(config['ftarget'], 'rb') as f_targets:
            self.targets = pickle.load(f_targets)
        with open(config['fcandidates'], 'rb') as f_candidates:
            self.candidates = pickle.load(f_candidates)

    def minimize(self, space, **min_args):
        minimizer = self.config['minimizer']
        return minimizer(self.objective, space, **min_args)

    def split_lex_mturk(self,test_size=0.4):
        """
        fdata: Path object
        """
        fstem = self.config['fdata'].stem
        fval = self.config['fdata'].parent / (fstem + '_val' + '.pickle')
        if not Path(fval).exists():
            print('prepare val test sentences')
            lines = self.config['fdata'].read_text()
            sens = [line.split() for line in lines.split('\n')[:-1]]
            X_val, X_test, val_targets, test_targets, val_candidates, test_candidates = \
                train_test_split(sens,self.targets,self.candidates, test_size=test_size)
            Xs = [[X_val,X_test],[val_targets,test_targets],[val_candidates,test_candidates]]
            fs,types = ['fdata','ftarget','fcandidates'],['val','test']
            for i,f in enumerate(fs):
                fstem = self.config[f].stem
                for j,type in enumerate(types):
                    f_type = self.config[f].parent / (fstem + f'_{type}' + '.pickle')
                    with open(f_type,'wb') as handle:
                        pickle.dump(Xs[i][j],handle,pickle.HIGHEST_PROTOCOL)
        else:
            print('load val test sentences')
            Xs = []
            fs, types = ['fdata', 'ftarget', 'fcandidates'], ['val', 'test']
            for i, f in enumerate(fs):
                fstem = self.config[f].stem
                stem = []
                for j, type in enumerate(types):
                    f_type = self.config[f].parent / (fstem + f'_{type}' + '.pickle')
                    with open(f_type, 'rb') as handle:
                        stem.append(pickle.load(handle))
                Xs.append(stem)

        return Xs

    def objective(self,feasible_point):
        parameters = {"complexity_drop_threshold": feasible_point[2], "num_cand": feasible_point[0],
                      "similarity_threshold": self.config['tholdsim'], "context_window_size": feasible_point[1],
                      "complexity_threshold": self.config['tholdcmplx']}
        simplifier = lightls.LightLS(self.embeddings, self.wfs, parameters, self.stopwords)
        simplifications = simplifier.simplify_lex_mturk(self.eval_data, self.eval_targets)
        acc, change = simplifier.evaluate_lex_mturk_simplification(simplifications, self.eval_candidates)
        return -acc

    def run(self):

        # split text
        Xs = self.split_lex_mturk()

        # optimize on the val data and test on the testing data
        self.eval_data, self.eval_targets, self.eval_candidates = [x[0] for x in Xs]
        x0 = [2,5,0.03]
        space = [
            Integer(2, 300),
            Integer(2, 10),
            Real(10 ** -5, 10 ** -1,'log-uniform')
        ]
        res = self.minimize(space, x0=x0, n_calls=50, verbose=True)

        self.eval_data, self.eval_targets, self.eval_candidates = [x[1] for x in Xs]
        print(f'test acc: {-self.objective(res.x)}')
        dump(res,'res-hyp.pickle',store_objective=False)

class SpLightLSExperiments(LightLSExperiments):

    def __init__(self,sp_model,dataset,config):
        self.config = config
        self.sp_model = sp_model
        self.dataset = dataset
        if self.config['exp_name'] == 'hrswe':
            emb = [vec for vec in dataset.emb_dict.values()]
            emb = np.vstack(emb).astype(np.float32).T

            # scaler = StandardScaler()
            # emb = scaler.fit_transform(emb.T).T

            # configuration: normalize vector
            # emb_norm = np.linalg.norm(emb, axis=0)[np.newaxis, :]
            # emb = emb / emb_norm
            # print(np.linalg.norm(emb,axis=0)[np.newaxis,:])
            d, n = emb.shape
            W = emb.T @ emb

            adj_pos, adj_neg, G = generate_syn_ant_graph(dataset.words, dataset.syn_pairs, dataset.ant_pairs)
            adj_spread = nx.adjacency_matrix(G, nodelist=dataset.words)

            self.model_kws = {
                'adj_pos': adj_pos,
                'adj_neg': adj_neg,
                'adj_spread': adj_spread,
                'W': W,
                'n': n,
                'd': d
            }
        elif self.config['exp_name'] == 'ar':
            self.model_kws = {}
        # preparation

        print("Loading unigram frequencies...")
        ls = io_helper.load_lines(config['fwordreqs'])
        self.wfs = {x.split()[0].strip(): int(x.split()[1].strip()) for x in ls}

        self.stopwords = io_helper.load_lines(config['fstopwords']) if config['fstopwords'] else None

        with open(config['ftarget'], 'rb') as f_targets:
            self.targets = pickle.load(f_targets)
        with open(config['fcandidates'], 'rb') as f_candidates:
            self.candidates = pickle.load(f_candidates)

    def sp_objective(self,feasible_point):
        model = self.sp_model(*feasible_point, **self.model_kws)
        sp_emb = model.specialize_emb(self.dataset.emb_dict,
                                      self.dataset.syn_pairs, self.dataset.ant_pairs)
        sp_emb_dict = {w: sp_emb[i, :] for i, w in enumerate(self.dataset.words)}

        self.embeddings = text_embeddings.PickleEmbedding()
        self.embeddings.load_embeddings(sp_emb_dict, self.config['word_limit'], language='default',
                                        print_loading=True, skip_first_line=False,
                                        normalize=True)

        res = self.minimize(self.config['ls_opt_space'], x0=self.config['ls_opt_x0'], n_calls=self.config['ls_n_calls'], verbose=False)
        return res.fun

    def sp_minimize(self, space, **min_args):
        minimizer = self.config['minimizer']
        return minimizer(self.sp_objective, space, **min_args)

    def run(self):

        # split text
        Xs = self.split_lex_mturk()

        # optimize on the val data and test on the testing data
        self.eval_data, self.eval_targets, self.eval_candidates = [x[0] for x in Xs]

        res = self.sp_minimize(self.config['sp_opt_space'], x0=self.config['sp_opt_x0'], n_calls=self.config['sp_n_calls'], verbose=True)

        self.eval_data, self.eval_targets, self.eval_candidates = [x[1] for x in Xs]
        print(f'test acc: {-self.sp_objective(res.x)}')
        dump(res,'res-hyp.pickle',store_objective=False)