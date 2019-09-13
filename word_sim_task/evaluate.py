from configparser import ConfigParser

from word_sim_task.local_utils import evaluate_similarity_on_tasks


class evaluator(object):

    def __init__(self,best_eval_score,tasks):
        self.best_eval_score = best_eval_score
        self.tasks = tasks

    def evaluate(self,embedding_dict,model):
        benchmark_scores = evaluate_similarity_on_tasks(self.tasks, embedding_dict)
        new_eval_score = sum(benchmark_scores.values())
        if new_eval_score > self.best_eval_score:
            model.print_word_vectors(model.word_vectors, model.output_filepath)
            self.best_eval_score = new_eval_score
            print('Current best eval score: %f' %(self.best_eval_score))
            print('Writing best parameters to cfg')

            with open('best_AR_parameters.cfg', 'w') as configfile:
                model.config.write(configfile)