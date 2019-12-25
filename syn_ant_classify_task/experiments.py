import pickle
from collections import defaultdict
from math import inf

import numpy as np

from evaluate import SynAntClyEvaluator


class BaseExperiments(object):

    def __init__(self,model,val_evaluator,test_evaluator,dataset,config):

        self.model = model
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self.dataset = dataset
        self.config = config # model_config(HRSWE config),hyp_tune_config(opt_space, bayes func),exp_config(save_emb,exp_name)

    def get_val_score(self, feasible_hyps):

        model = self.model(*feasible_hyps)
        sp_emb = model.specialize_emb(self.dataset.emb_dict,
                                      self.dataset.syn_pairs,self.dataset.ant_pairs)
        sp_emb_dict = {w:sp_emb[i,:] for i,w in enumerate(self.dataset.words)}
        score,_ = self.val_evaluator.eval_emb_on_tasks(sp_emb_dict)
        self.val_evaluator.update_results()

        return -score # minimize to optimize

    def run(self):

        hyp_tune_func = self.config['hyp_tune_func']
        res = hyp_tune_func(self.get_val_score, self.config['hyp_opt_space'],
                                 **self.config['tune_func_config'])

        best_emb_dict = self.val_evaluator.best_emb
        score, test_res = self.test_evaluator.eval_emb_on_tasks(best_emb_dict)

        if self.config['exp_config']['save_res']:
            final_res = {
                'best_emb_dict':best_emb_dict,
                'test_res':test_res,
                'best_val_res':self.val_evaluator.best_results,
                'best_hyps':res.x,
                'config':self.config
            }
            if isinstance(self.test_evaluator,SynAntClyEvaluator):
                final_res['best_th'] = test_res['th']

            with open(self.config['exp_config']['exp_name']+'_results' + '.pickle', 'wb') as handle:
                pickle.dump(final_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return score

    def run_InjectedMatrix(self,*hyps):
        # todo: refactor this function latter
        beta1s, beta2s = hyps

        words_emb = [self.dataset.emb_dict[w] for w in self.dataset.words]
        words2id = {w:i for i,w in enumerate(self.dataset.words)}
        words_emb = np.vstack(words_emb).T
        # todo: normalize or not?
        # words_emb = words_emb / linalg.norm(words_emb,axis=0)

        results = defaultdict(list)
        results['beta_range1'] = beta1s
        results['beta_range2'] = beta2s

        cur_best_score = -np.inf
        adj_pos, adj_neg = self.dataset.generate_syn_ant_graph()

        W = words_emb.T @ words_emb

        for beta1 in beta1s:
            for beta2 in beta2s:

                W_prime = W + beta1 * adj_pos.multiply(np.max(W) - W) + beta2 * adj_neg.multiply(
                    W - np.min(W))

                # check this matrix with validation data
                scores = self.evaluator.eval_injected_matrix({"SIMVERB500-dev": self.dataset.sim_tasks["SIMVERB500-dev"]},
                                                        W_prime,words2id)

                if scores['SIMVERB500-dev'] > cur_best_score:

                    results['best_matrix'] = W_prime
                    results['best_betas'] = [beta1,beta2]
                    cur_best_score = scores['SIMVERB500-dev']

        self.evaluator.eval_injected_matrix(self.dataset.sim_tasks,results['best_matrix'],words2id)