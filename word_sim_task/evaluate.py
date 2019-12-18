from configparser import ConfigParser

import scipy
from six import iteritems
from web.evaluate import evaluate_similarity


class evaluator(object):

    def __init__(self,best_eval_score,tasks):
        self.best_eval_score = best_eval_score
        self.tasks = tasks

    def eval_AR(self, embedding_dict, model):
        benchmark_scores = self.eval_emb_on_sim(self.tasks, embedding_dict)
        new_eval_score = sum(benchmark_scores.values())
        if new_eval_score > self.best_eval_score:
            # save specialized vectors
            model.print_word_vectors(model.word_vectors, model.output_filepath)
            self.best_eval_score = new_eval_score
            print('Current best eval score: %f' %(self.best_eval_score))
            print('Writing best parameters to cfg')
            model.config.set('hyperparameters','curr_iter',model.current_iteration)
            with open('best_AR_parameters.cfg', 'w') as configfile:
                model.config.write(configfile)

    def eval_emb_on_sim(self, tasks, emb_obj):
        scores = {}
        print('*' * 30)
        for name, data in iteritems(tasks):
            score = evaluate_similarity(emb_obj, data.X, data.y)
            print("Spearman correlation of scores on {} {}".format(name, score))
            scores[name] = score
        return scores

    def eval_injected_matrix(self,tasks,matrix,words2id):
        scores = {}
        print('*' * 30)
        for name, data in iteritems(tasks):
            pred_y = [matrix[tuple(words2id[w] for w in p)] for p in data.X]
            score = scipy.stats.spearmanr(pred_y, data.y).correlation
            print("Spearman correlation of scores on {} {}".format(name, score))
            scores[name] = score
        return scores