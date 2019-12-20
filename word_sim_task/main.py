import configparser
import os
import shutil
from math import inf
from os.path import join
from pathlib import Path

import numpy as np
from constants import AR_THES_DIR, ORIGINAL_VECS_DIR, ATTRACT_REPEL_DIR, ATTRACT_REPEL_VECS, WORD_SIM_TASK_DIR
from word_sim_task.dataset import Dataset
from word_sim_task.evaluate import evaluator
from word_sim_task.experiments import BaseExperiments
from word_sim_task.model import HRSWE
from attract_repel.attract_repel import ExperimentRun as AR

# prepare dataset
adv_thesauri = {'syn_fname': join(AR_THES_DIR, 'adv_sub_synonyms.txt'),
                    'ant_fname': join(AR_THES_DIR, 'adv_sub_antonyms.txt')}
ori_thesauri = {'syn_fname': join(AR_THES_DIR, 'sub_synonyms.txt'),
                'ant_fname': join(AR_THES_DIR, 'sub_antonyms.txt')}
dataset = Dataset()
dataset.load_datasets()
dataset.load_words()
dataset.load_embeddings()

# AR config
config_filepath = 'AR_parameters.cfg'
ar_config = configparser.RawConfigParser()
ar_config.read(config_filepath)
sel_vec_fname = join(ORIGINAL_VECS_DIR, dataset.vocab_fname+'.pickle')
ar_config.set('data', 'distributional_vectors', sel_vec_fname)
antonyms_list = [adv_thesauri['ant_fname']]
synonyms_list = [adv_thesauri['syn_fname']]
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
    'thesauri': adv_thesauri,
    'sim_mat_type':'pd',
    'eig_vec_option':'ld',
    'emb_type':0
}

# HRSWE hyps
beta1s = np.linspace(0,1,21)
beta2s = np.linspace(0,1,21)


# exp.run_InjectedMatrix(beta1s,beta2s)

for r in [0.1,0.4,0.5,0.6,0.7,0.8,0.9]:
# for r in [0.2]:
    # prepare dataset
    dataset.generate_adv_thesaurus2(r,ori_thesauri)

    # evaluator
    evaluator_obj = evaluator(best_eval_score=-inf,tasks={"SIMVERB500-dev":dataset.sim_tasks["SIMVERB500-dev"]})

    # experiments
    exp = BaseExperiments(dataset, HRSWE, AR, evaluator_obj)

    # run exps
    hrswe_results = exp.run_HRSWE(beta1s,beta2s,**hrswe_config)
    ar_results = exp.run_AR(ar_config,synonym_margins,antonym_margins)
    np.save('hrswe_results_'+str(r),hrswe_results)
    np.save('ar_results_'+str(r),ar_results)


    # move files to dirs
    ta_dir = join(WORD_SIM_TASK_DIR,'results','adv_results','adv2','inter_'+str(r))
    # create results dir
    Path(ta_dir).mkdir(parents=True, exist_ok=True)
    # move relevant files into results dir
    shutil.move('best_AR_parameters.cfg', join(ta_dir,'best_AR_parameters.cfg'))
    shutil.move('results.txt', join(ta_dir, 'results.txt'))
    shutil.move('wn_ro_pd_ld_0.pickle', join(ta_dir, 'wn_ro_pd_ld_0.pickle'))
    shutil.move(join(ATTRACT_REPEL_VECS, dataset.vocab_fname+'.pickle'),join(ta_dir,dataset.vocab_fname+'.pickle'))
    # create adv thesauri dir and move adv thesauri to that dir
    ta_dir = join(AR_THES_DIR,'adv2','inter_'+str(r))
    Path(ta_dir).mkdir(parents=True, exist_ok=True)
    shutil.move(adv_thesauri['syn_fname'],join(ta_dir,'adv_sub_synonyms.txt'))
    shutil.move(adv_thesauri['ant_fname'],join(ta_dir,'adv_sub_antonyms.txt'))

# model = AR(ar_config)
# model.attract_repel(evaluator)
