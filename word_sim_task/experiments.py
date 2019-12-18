import pickle
import time
from collections import defaultdict

from scipy import linalg
from web.embedding import Embedding
import numpy as np
# import cvxpy as cp
import plotly.graph_objs as go
from plotly.offline import plot


class BaseExperiments(object):

    def __init__(self, dataset, HRSWE, AR, evaluator):
        self.dataset = dataset
        self.HRSWE = HRSWE
        self.AR = AR
        self.evaluator = evaluator

    def run_HRSWE(self, *hyps, **config):
        beta1s,beta2s = hyps
        self.results_fname = '_'.join([config['thesauri_name'], config['sim_mat_type'], config['eig_vec_option'], str(config['emb_type'])])

        # betas = [0.5]
        # beta1s = [10,50,100]
        # beta2s = [10,50,100]
        # word_sim_pairs = combine_bunches(*sim_tasks.values())
        # thesauri = {'name':thesauri_name,'word_sim_pairs':word_sim_pairs}

        words_emb = [self.dataset.emb_dict[w] for w in self.dataset.words]
        words_emb = np.vstack(words_emb).T
        # words_emb = words_emb / linalg.norm(words_emb,axis=0)

        results = defaultdict(list)
        results['beta_range1'] = beta1s
        results['beta_range2'] = beta2s

        cur_best_score = -np.inf
        adj_pos, adj_neg = self.dataset.generate_syn_ant_graph()
        times = []
        for beta1 in beta1s:
            for beta2 in beta2s:
                last_time = time.time()
                model = self.HRSWE(beta1, beta2, words_emb, adj_pos, adj_neg, config)
                emb = model.specialize_emb()
                time_spend = round(time.time() - last_time, 1)
                times.append(time_spend)
                print('Time took: ', time_spend)
                emb_obj = Embedding.from_dict({w: emb[:, i] for i, w in enumerate(self.dataset.words)})
                scores = self.evaluator.eval_emb_on_sim({"SIMVERB500-dev": self.dataset.sim_tasks["SIMVERB500-dev"]}, emb_obj)
                for k, v in scores.items():
                    results[k + '_scores'].append(v)
                summed_score = sum(scores.values())
                results['summed_score'].append(summed_score)
                # save current best embeddings
                if summed_score > cur_best_score:
                    cur_best_score = summed_score
                    results['best_summed_scores'] = cur_best_score
                    results['best_scored_emb'] = emb_obj
                    results['best_betas'] = [beta1,beta2]
        print('Average time spent: ', round(sum(times) / len(times), 1))

        with open(self.results_fname + '.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.evaluator.eval_emb_on_sim(self.dataset.sim_tasks, results["best_scored_emb"])

    def run_InjectedMatrix(self,*hyps):

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

    def run_AR(self, config, *hyps):

        synonym_margins,antonym_margins = hyps
        for s_m in synonym_margins:
            for a_m in antonym_margins:
                config.set('hyperparameters', 'attract_margin', s_m)
                config.set('hyperparameters', 'repel_margin', a_m)

                model = self.AR(config)
                model.attract_repel(self.evaluator)

        # eval the ar specialized embedding
        with open(config.get('data','output_filepath'), 'rb') as handle:
            emb_dict = pickle.load(handle)
        self.evaluator.eval_emb_on_sim(self.dataset.sim_tasks,emb_dict)

    def draw_HRSWE_dev_results(self):

        # results['benchmark_scores'] = benchmark_scores
        with open(self.results_fname + '.pickle', 'rb') as handle:
            results = pickle.load(handle)
        self.evaluator.eval_emb_on_sim(self.dataset.sim_tasks, results["best_scored_emb"])

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

        # for name in task_names:
        #     benchmark_scores_trace.append(
        #         go.Scatter(
        #             x=results['beta_range'],
        #             y=[results['benchmark_scores'][name]] * len(results['beta_range']),
        #             mode='lines+markers',
        #             name='SGNS-GN'
        #         )
        #     )

        plot({
            "data": results_trace,
            # "data": results_trace + benchmark_scores_trace,
            "layout": go.Layout(),
        }, filename=self.results_fname + '.html')