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

class MatrixExperiments(BaseExperiments):

    def get_val_score(self, feasible_hyps):

        model = self.model(*feasible_hyps)
        rf_matrix = model.specialize_emb(self.dataset.emb_dict,
                                      self.dataset.syn_pairs,self.dataset.ant_pairs)
        word2id = {w:i for i,w in enumerate(self.dataset.words)}
        score,_ = self.val_evaluator.eval_injected_matrix(rf_matrix,word2id)
        self.val_evaluator.update_results()

        return -score # minimize to optimize

    def run(self):

        hyp_tune_func = self.config['hyp_tune_func']
        res = hyp_tune_func(self.get_val_score, self.config['hyp_opt_space'],
                                 **self.config['tune_func_config'])

        best_emb_dict = self.val_evaluator.best_emb
        word2id = {w: i for i, w in enumerate(self.dataset.words)}
        score, test_res = self.test_evaluator.eval_injected_matrix(best_emb_dict,word2id)

        if self.config['exp_config']['save_res']:
            final_res = {
                'best_matrix':best_emb_dict,
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