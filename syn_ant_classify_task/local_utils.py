from math import inf

import numpy as np


class BaseExperiments(object):

    def __init__(self,model,val_evaluator,test_evaluator,dataset,hyp_tune_func,config):

        self.hyp_tune_func = hyp_tune_func
        self.model = model
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self.dataset = dataset
        self.config = config # model_config,hyp_tune_config(opt_space),exp_config(save_emb)

    def get_val_score(self, feasible_point):

        model = self.model(*feasible_point,**self.config['model_config'])
        sp_emb = model.specialize_emb(self.dataset.emb)
        sp_emb_dict = {w:sp_emb[i,:] for i,w in enumerate(self.dataset.words)}
        score,_ = self.val_evaluator.eval_emb_on_tasks(sp_emb_dict)
        self.val_evaluator.update_results()

        return score

    def run_exp(self):

        res = self.hyp_tune_func(self.get_val_score, self.config['hyp_tune_config']['opt_space'],
                                 **self.config['hyp_tune_config'])

        best_emb_dict = self.val_evaluator.best_results['emb_dict']
        score, _ = self.test_evaluator.eval_emb_on_tasks(best_emb_dict)

        # todo: save results
