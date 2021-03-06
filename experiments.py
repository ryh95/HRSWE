import pickle
import time
from collections import defaultdict, OrderedDict
from math import inf

import networkx as nx
import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

from evaluate import SynAntClyEvaluator
from model import generate_syn_ant_graph, generate_spread_graph, generate_spread_graph_1, generate_spread_graph_2, \
    generate_spread_graph_3, generate_spread_graph_4, generate_spread_graph_5


class BaseExperiments(object):

    def __init__(self,model,val_evaluator,test_evaluator,dataset,config):

        self.model = model
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self.dataset = dataset
        self.config = config # model_config(HRSWE config),hyp_tune_config(opt_space, bayes func),exp_config(save_emb,exp_name)

    def get_val_score(self, feasible_hyps):

        model = self.model(*feasible_hyps,**self.model_kws)
        sp_emb = model.specialize_emb(self.dataset.emb_dict,
                                      self.dataset.syn_pairs,self.dataset.ant_pairs)
        sp_emb_dict = {w:sp_emb[i,:] for i,w in enumerate(self.dataset.words)}
        score,_ = self.val_evaluator.eval_emb_on_tasks(sp_emb_dict)
        self.val_evaluator.update_results()

        # minimize to optimize, 10 is used to handle exception
        return -score if not np.isnan(score) else 10

    def obtain_best_emb(self):

        hyp_tune_func = self.config['hyp_tune_func']
        start = time.time()
        res = hyp_tune_func(self.get_val_score, self.config['hyp_opt_space'],
                            **self.config['tune_func_config'])

        self.tune_time = time.time() - start
        self.best_hyps = res.x

    def test_best_emb(self):

        best_emb_dict = self.val_evaluator.best_emb
        self.test_evaluator.eval_emb_on_tasks(best_emb_dict)

    def pack_dump_res(self):
        if self.config['exp_config']['save_res']:
            final_emb = {
                'best_emb_dict':self.test_evaluator.cur_emb
            }
            final_res = {
                'test_res':self.test_evaluator.cur_results,
                'best_val_res':self.val_evaluator.best_results,
                'best_hyps':self.best_hyps,
                'tune_time':self.tune_time,
                'config':self.config
            }
            if isinstance(self.test_evaluator,SynAntClyEvaluator):
                final_res['best_th'] = self.test_evaluator.cur_results['th']
            with open(self.config['exp_config']['exp_name']+'_emb' + '.pickle', 'wb') as handle:
                pickle.dump(final_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(self.config['exp_config']['exp_name']+'_results' + '.pickle', 'wb') as handle:
                pickle.dump(final_res, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def run(self):

        self.obtain_best_emb()
        self.test_best_emb()
        self.pack_dump_res()

class HRSWEExperiments(BaseExperiments):

    def __init__(self,model,val_evaluator,test_evaluator,dataset,config):
        super().__init__(model,val_evaluator,test_evaluator,dataset,config)

        emb = [vec for vec in dataset.emb_dict.values()]
        emb = np.vstack(emb).astype(np.float32).T

        scaler = StandardScaler()
        emb = scaler.fit_transform(emb.T).T

        # configuration: normalize vector
        # emb_norm = np.linalg.norm(emb, axis=0)[np.newaxis, :]
        # emb = emb / emb_norm
        # print(np.linalg.norm(emb,axis=0)[np.newaxis,:])
        d,n = emb.shape
        W = emb.T @ emb

        adj_pos,adj_neg,G = generate_syn_ant_graph(dataset.words,dataset.syn_pairs,dataset.ant_pairs)
        adj_spread = nx.adjacency_matrix(G, nodelist=dataset.words)

        # G_spread = generate_spread_graph(G,0.4,0.6)
        # adj_spread = nx.adjacency_matrix(G_spread, nodelist=dataset.words)
        # self.model_kws = {
        #     'adj_pos':adj_pos,
        #     'adj_neg':adj_neg,
        #     'adj_spread':adj_spread
        # }
        # G_spread = generate_spread_graph_5(G)
        # adj_spread = nx.adjacency_matrix(G_spread, nodelist=dataset.words)

        # u,s,vt = svd(adj_pos.toarray())
        # explain_ratio = np.cumsum(s ** 2) / np.sum(s ** 2)
        # sel_rank = np.argwhere(explain_ratio>0.9)[0][0]
        #
        # adj_pos = u[:,:sel_rank] * s[:sel_rank] @ vt[:sel_rank,:]
        # # #
        # u, s, vt = svd(adj_neg.toarray())
        # explain_ratio = np.cumsum(s ** 2) / np.sum(s ** 2)
        # sel_rank = np.argwhere(explain_ratio > 0.9)[0][0]
        #
        # adj_neg = u[:, :sel_rank] * s[:sel_rank] @ vt[:sel_rank, :]
        #
        # u,s,vt = svd(adj_spread.toarray())
        # explain_ratio = np.cumsum(s ** 2) / np.sum(s ** 2)
        # sel_rank = np.argwhere(explain_ratio>0.9)[0][0]
        #
        # adj_spread = u[:,:sel_rank] * s[:sel_rank] @ vt[:sel_rank,:]

        self.model_kws = {
            'adj_pos': adj_pos,
            'adj_neg': adj_neg,
            'adj_spread': adj_spread,
            'W':W,
            'n':n,
            'd':d
        }


class ARExperiments(BaseExperiments):

    def __init__(self,model,val_evaluator,test_evaluator,dataset,config):
        super().__init__(model,val_evaluator,test_evaluator,dataset,config)

        # standardize the embedding
        # emb = [vec for vec in dataset.emb_dict.values()]
        # emb = np.vstack(emb)
        #
        # scaler = StandardScaler()
        # emb = scaler.fit_transform(emb)
        #
        # self.dataset.emb_dict = OrderedDict((w,emb[i,:]) for i,w in enumerate(dataset.emb_dict.keys()))

        self.model_kws = {}

class MatrixExperiments(BaseExperiments):

    def __init__(self,model,val_evaluator,test_evaluator,dataset,config):
        super().__init__(model,val_evaluator,test_evaluator,dataset,config)

        emb = [vec for vec in dataset.emb_dict.values()]
        emb = np.vstack(emb).astype(np.float32).T

        # configuration: normalize vector
        emb_norm = np.linalg.norm(emb, axis=0)[np.newaxis, :]
        emb = emb / emb_norm
        # print(np.linalg.norm(emb,axis=0)[np.newaxis,:])

        W = emb.T @ emb

        adj_pos,adj_neg,G = generate_syn_ant_graph(dataset.words,dataset.syn_pairs,dataset.ant_pairs)
        # G_spread = generate_spread_graph_4(G,dataset)
        # adj_spread = nx.adjacency_matrix(G_spread, nodelist=dataset.words)
        adj_spread = nx.adjacency_matrix(G,nodelist=dataset.words)
        self.model_kws = {
            'adj_pos':adj_pos,
            'adj_neg':adj_neg,
            'adj_spread':adj_spread,
            'W':W,
        }

    def get_val_score(self, feasible_hyps):

        model = self.model(*feasible_hyps,**self.model_kws)
        rf_matrix = model.specialize_emb(self.dataset.emb_dict,
                                      self.dataset.syn_pairs,self.dataset.ant_pairs)
        word2id = {w:i for i,w in enumerate(self.dataset.words)}
        score,_ = self.val_evaluator.eval_injected_matrix(rf_matrix,word2id)
        self.val_evaluator.update_results()

        return -score # minimize to optimize

    def test_best_emb(self):
        best_emb_dict = self.val_evaluator.best_emb
        word2id = {w: i for i, w in enumerate(self.dataset.words)}
        self.test_evaluator.eval_injected_matrix(best_emb_dict, word2id)