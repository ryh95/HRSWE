import configparser
from math import inf
from os.path import join
import numpy as np
from constants import AR_THES_DIR, ORIGINAL_VECS_DIR, ATTRACT_REPEL_DIR, ATTRACT_REPEL_VECS
from word_sim_task.dataset import Dataset
from word_sim_task.evaluate import evaluator
from word_sim_task.experiments import BaseExperiments
from word_sim_task.model import HRSWE
from attract_repel.attract_repel import ExperimentRun as AR

thesauri = {'syn_fname': join(AR_THES_DIR, 'sub_synonyms.txt'),
                    'ant_fname': join(AR_THES_DIR, 'sub_antonyms.txt')}
# prepare dataset
dataset = Dataset(thesauri)
dataset.load_datasets()
dataset.load_words()
dataset.load_embeddings()

# AR config
config_filepath = 'AR_parameters.cfg'
ar_config = configparser.RawConfigParser()
ar_config.read(config_filepath)
sel_vec_fname = join(ORIGINAL_VECS_DIR, dataset.vocab_fname+'.pickle')
ar_config.set('data', 'distributional_vectors', sel_vec_fname)
antonyms_list = [thesauri['syn_fname']]
synonyms_list = [thesauri['ant_fname']]
ar_config.set('data', 'antonyms_list', antonyms_list)
ar_config.set('data', 'synonyms_list', synonyms_list)
ar_config.set('data', 'eval_dir_path', join(ATTRACT_REPEL_DIR, 'train_eval_data'))
ar_config.set('data', 'output_filepath', join(ATTRACT_REPEL_VECS, dataset.vocab_fname+'.pickle'))

# AR hyps
synonym_margins = np.linspace(0,1,11)
antonym_margins = np.linspace(0,1,11)

# HRSWE config
hrswe_config = {
    'thesauri_name':'wn_ro',
    'sim_mat_type':'pd',
    'eigen_vec_option':'ld',
    'emb_type':0
}

# HRSWE hyps
beta1s = np.linspace(0,1,21)
beta2s = np.linspace(0,1,21)

# evaluator
evaluator = evaluator(best_eval_score=-inf,tasks={"SIMVERB500-dev":dataset.sim_tasks["SIMVERB500-dev"]})

# experiments
exp = BaseExperiments(dataset, HRSWE, AR, evaluator)
exp.run_HRSWE(beta1s,beta2s,**hrswe_config)
exp.run_AR(ar_config,synonym_margins,antonym_margins)
