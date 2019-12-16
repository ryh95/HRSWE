import configparser
import os
import pickle
from math import inf
from os.path import join
import numpy as np
from web.datasets.similarity import fetch_SimLex999

from attract_repel.attract_repel import run_experiment, ExperimentRun
from constants import ATTRACT_REPEL_DIR, ORIGINAL_VECS_DIR, AR_THES_DIR, ATTRACT_REPEL_VECS, WORD_SIM_TASK_DIR

# config_filepath = join(ATTRACT_REPEL_DIR,"AR_parameters.cfg")
from preprocess import GeneralTextProcesser
from word_sim_task.evaluate import evaluator
from word_sim_task.local_utils import evaluate_similarity_on_tasks, fetch_SimVerb3500

config_filepath = 'AR_parameters.cfg'

config = configparser.RawConfigParser()
try:
    config.read(config_filepath)
except:
    assert False, "Couldn't read config file from %s" % (config_filepath)

sel_vec_fname = join(ORIGINAL_VECS_DIR,'SIMLEX999_SIMVERB3000-test_SIMVERB500-dev.pickle')
config.set('data','distributional_vectors',sel_vec_fname)
antonyms_list = [join(AR_THES_DIR,'sub_antonyms.txt')]
synonyms_list = [join(AR_THES_DIR,'sub_synonyms.txt')]
config.set('data','antonyms_list',antonyms_list)
config.set('data','synonyms_list',synonyms_list)
config.set('data','eval_dir_path',join(ATTRACT_REPEL_DIR,'train_eval_data'))
config.set('data','output_filepath',join(ATTRACT_REPEL_VECS,'SIMLEX999_SIMVERB3000-test_SIMVERB500-dev.pickle'))

synonym_margins = np.linspace(0,1,11)
antonym_margins = np.linspace(0,1,11)


sim_tasks = {
            "SIMLEX999": fetch_SimLex999(),
            "SIMVERB3000-test" : fetch_SimVerb3500(join(WORD_SIM_TASK_DIR,'task_data','SimVerb-3000-test.txt')),
            "SIMVERB500-dev" : fetch_SimVerb3500(join(WORD_SIM_TASK_DIR,'task_data','SimVerb-500-dev-adv.txt')),
        }

ar_evaluator = evaluator(best_eval_score=-inf,tasks={"SIMVERB500-dev":sim_tasks["SIMVERB500-dev"]})

for s_m in synonym_margins:
    for a_m in antonym_margins:
        config.set('hyperparameters','attract_margin',s_m)
        config.set('hyperparameters','repel_margin',a_m)
        config.set('hyperparameters', 'max_iter', 10)

        model = ExperimentRun(config)
        model.attract_repel(ar_evaluator)
