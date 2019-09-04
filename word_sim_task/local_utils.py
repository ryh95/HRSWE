import time
from collections import defaultdict

import numpy as np
from six import iteritems
from sklearn.utils import Bunch
from web.embedding import Embedding
import plotly.graph_objs as go
from plotly.offline import plot
from web.evaluate import evaluate_similarity

from utils import prepare_syn_ant_graph, DAWE


def fetch_SimVerb3500(simverb_fname):
    X,y = [],[]
    with open(simverb_fname,'r') as f:
        for line in f:
            try:
                w1,w2,_,score,_ = line.strip().split('\t')
            except ValueError:
                w1, w2, _, score = line.strip().split('\t')
            X.append([w1,w2])
            y.append(score)
    X = np.vstack(X)
    y = np.array(y,dtype=float)
    return Bunch(X=X.astype("object"),y=y)

def get_all_words_in_constraint(word_sim_fname):
    vocab = set()
    with open(word_sim_fname, 'r') as f:
        for line in f:
            word_pair = line.split()
            word_pair = [word[3:] for word in word_pair]  # remove the 'en-' prefix
            vocab.update(word_pair)
    return vocab

def create_test_emb_on_word_sim_tasks(words,words_vecs,sim_tasks,beta1s,beta2s,
                                      thesauri,sim_type,eigen_vec_option,emb_type):

    config = {'sim_mat_type': sim_type, 'eig_vec_option': eigen_vec_option, 'emb_type': emb_type}
    results = defaultdict(list)
    results['beta_range1'] = beta1s
    results['beta_range2'] = beta2s

    cur_best_score = -np.inf
    adj_pos, adj_neg = prepare_syn_ant_graph(words, thesauri)
    times = []
    for beta1 in beta1s:
        for beta2 in beta2s:
            last_time = time.time()
            emb = DAWE(beta1,beta2, words_vecs, adj_pos,adj_neg, config)
            time_spend = round(time.time() - last_time, 1)
            times.append(time_spend)
            print('Time took: ', time_spend)
            emb_obj = Embedding.from_dict({w: emb[:, i] for i, w in enumerate(words)})
            scores = evaluate_similarity_on_tasks(sim_tasks, emb_obj)
            for k, v in scores.items():
                results[k + '_scores'].append(v)
            summed_score = sum(scores.values())
            results['summed_score'].append(summed_score)
            # save current best embeddings
            if summed_score > cur_best_score:
                cur_best_score = summed_score
                results['best_summed_scores'] = cur_best_score
                results['best_scored_emb'] = emb_obj
    print('Average time spent: ', round(sum(times) / len(times), 1))
    return results

def draw_results(results, fname, task_names):
    results_trace = []
    # benchmark_scores_trace = []
    x_grid,y_grid = np.meshgrid(results['beta_range1'],results['beta_range2'])
    n = results['beta_range1'].size
    for name in task_names:
        z_grid = np.array(results[name + '_scores']).reshape(n,n).T
        results_trace.append(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                # mode='lines+markers',
                name='LHRSWE' # 'HRSWE'
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
        "data" : results_trace,
        # "data": results_trace + benchmark_scores_trace,
        "layout": go.Layout(),
    },filename=fname)

def evaluate_similarity_on_tasks(tasks,emb_obj):
    scores = {}
    print('*'*30)
    for name, data in iteritems(tasks):
        score = evaluate_similarity(emb_obj, data.X, data.y)
        print("Spearman correlation of scores on {} {}".format(name, score))
        scores[name] = score
    return scores