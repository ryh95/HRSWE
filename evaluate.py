import pickle
import sys
from math import inf

import numpy as np
import scipy
from joblib import Parallel, delayed
from six import iteritems
from sklearn.metrics import precision_recall_fscore_support
from web.evaluate import evaluate_similarity
import plotly.graph_objs as go
from plotly.offline import plot
from abc import ABC, abstractmethod

class Evaluator(ABC):

    def __init__(self,tasks):
        self.cur_score = None
        self.cur_results = {}
        self.best_score = -inf
        self.best_results = {}
        self.cur_emb = None
        self.best_emb = None
        self.tasks = tasks

    def update_results(self):
        if self.cur_score > self.best_score:
            self.best_score = self.cur_score
            self.best_results = self.cur_results
            self.best_emb = self.cur_emb
            print('Current best eval score: %f' % (self.best_score))


    @abstractmethod
    def eval_emb_on_tasks(self, emb_dict, file):
        pass



class WordSimEvaluator(Evaluator):

    def eval_emb_on_tasks(self, emb_dict, file=sys.stdout):
        results = {}
        print('*' * 30)
        for name, data in self.tasks.items():
            score = evaluate_similarity(emb_dict, data.X, data.y)
            print("Spearman correlation of scores on {} {}".format(name, score),file=file)
            results[name] = score
        self.cur_score = sum(results.values())
        self.cur_results = results
        self.cur_emb = emb_dict
        return self.cur_score, self.cur_results

    def eval_injected_matrix(self,matrix,words2id, file=sys.stdout):
        results = {}
        print('*' * 30)
        for name, data in self.tasks.items():
            pred_y = [matrix[tuple(words2id[w] for w in p)] for p in data.X]
            score = scipy.stats.spearmanr(pred_y, data.y).correlation
            print("Spearman correlation of scores on {} {}".format(name, score),file=file)
            results[name] = score
        self.cur_score = sum(results.values())
        self.cur_results = results
        self.cur_emb = matrix
        return self.cur_score, self.cur_results

    def draw_HRSWE_dev_results(self,results_fname,sim_tasks):

        # results['benchmark_scores'] = benchmark_scores
        with open(results_fname + '.pickle', 'rb') as handle:
            results = pickle.load(handle)
        self.eval_emb_on_tasks(sim_tasks, results["best_scored_emb"])

        results_trace = []
        # benchmark_scores_trace = []
        x_grid, y_grid = np.meshgrid(results['beta_range1'], results['beta_range2'])
        n = results['beta_range1'].size
        for name in ["SIMVERB500-dev"]:
            z_grid = np.array(results[name + '_scores']).reshape(n, n).T
            results_trace.append(
                go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=z_grid,
                    # mode='lines+markers',
                    name='LHRSWE'  # 'HRSWE'
                )
            )

        plot({
            "data": results_trace,
            # "data": results_trace + benchmark_scores_trace,
            "layout": go.Layout(),
        }, filename='results' + '.html')

class SynAntClyEvaluator(Evaluator):

    def __init__(self,tasks,ths):
        super().__init__(tasks)
        self.ths = ths

    def eval_emb_on_tasks(self, emb_dict, f=sys.stdout):

        list_results = Parallel(n_jobs=10)(delayed(self.__eval_emb_with_th)(emb_dict, th) for th in self.ths)
        # save and print scores that have the highest total f1
        best_results = max(list_results,key=lambda x: x[0])
        self.cur_score = best_results[0]
        self.cur_results = best_results[1]
        self.cur_emb = emb_dict
        for k,v in self.cur_results.items():
            print('%s: %f' % (k, v), file=f)
        return self.cur_score, self.cur_results

    def __eval_emb_with_th(self, emb_dict, th):
        results = {}
        results['th'] = th
        total_f1 = 0
        for name, data in self.tasks.items():
            A = np.vstack([emb_dict[word] for word in data.X[:, 0]])
            B = np.vstack([emb_dict[word] for word in data.X[:, 1]])
            sim = np.array([v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)) for v1, v2 in zip(A, B)])

            y_preds = (sim < th).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(data.y, y_preds, average='binary')
            results[name + '_p'] = p
            results[name + '_r'] = r
            results[name + '_f1'] = f1
            total_f1 += f1
        results['total_f1'] = total_f1
        return total_f1,results