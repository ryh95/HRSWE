import shutil
from os.path import join
from pathlib import Path

import numpy as np
from constants import THESAURUS_DIR
from dataset import Dataset
from model import AR, HRSWE
from syn_ant_classify_task.config import ori_thesauri, adv_thesauri, ths, ar_config
from syn_ant_classify_task.experiments import BaseExperiments
from utils import generate_adv3, generate_sub_thesauri
from evaluate import Evaluator, SynAntClyEvaluator


# load datasets
dataset = Dataset()
dataset.load_task_datasets(['adjective-pairs.val','noun-pairs.val','verb-pairs.val',
                            'adjective-pairs.test','noun-pairs.test','verb-pairs.test'])
dataset.load_words()
dataset.load_embeddings()


val_tasks = {name:task for name,task in dataset.tasks.items() if 'val' in name}
generate_sub_thesauri(join(THESAURUS_DIR, 'synonyms.txt'),ori_thesauri['syn_fname'],set(dataset.words))
generate_sub_thesauri(join(THESAURUS_DIR, 'antonyms.txt'),ori_thesauri['ant_fname'],set(dataset.words))

for r in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    # prepare adv thesauri
    generate_adv3(r, ori_thesauri, dataset.tasks)
    dataset.load_thesauri(adv_thesauri)

    # evaluators
    hrswe_val = SynAntClyEvaluator(val_tasks, ths)
    hrswe_test = SynAntClyEvaluator(dataset.tasks, ths)
    ar_val = SynAntClyEvaluator(val_tasks, ths)
    ar_test = SynAntClyEvaluator(dataset.tasks,ths)

    # experiments
    ar_exp = BaseExperiments(AR,ar_val,ar_test,dataset,ar_config)
    hrswe_exp = BaseExperiments(HRSWE,hrswe_val,hrswe_test,dataset,ar_config)

    # run exps
    ar_test_score = ar_exp.run()
    ar_test_fscore = ar_exp.config['exp_config']['exp_name'] + '_score' + '.npy'
    np.save(ar_test_fscore, ar_test_score)

    hrswe_test_score = hrswe_exp.run()
    hrswe_test_fscore = hrswe_exp.config['exp_config']['exp_name'] + '_score' + '.npy'
    np.save(hrswe_test_fscore, hrswe_test_score)

    # move files to dirs
    ta_dir = join('results','adv_results','adv3',str(r))
    # create results dir
    Path(ta_dir).mkdir(parents=True, exist_ok=True)
    # move relevant files into results dir
    shutil.move(ar_test_fscore, join(ta_dir, ar_test_fscore))
    shutil.move(hrswe_test_fscore, join(ta_dir, hrswe_test_fscore))

    ar_fresults = ar_exp.config['exp_config']['exp_name'] + '_results' + '.pickle'
    shutil.move(ar_fresults, join(ta_dir, ar_fresults))
    hrswe_fresults = hrswe_exp.config['exp_config']['exp_name'] + '_results' + '.pickle'
    shutil.move(hrswe_fresults, join(ta_dir, hrswe_fresults))

    # create adv thesauri dir and move adv thesauri to that dir
    ta_dir = join(THESAURUS_DIR,'clf','adv3',str(r))
    Path(ta_dir).mkdir(parents=True, exist_ok=True)
    shutil.move(adv_thesauri['syn_fname'],join(ta_dir,'adv_synonyms.txt'))
    shutil.move(adv_thesauri['ant_fname'],join(ta_dir,'adv_antonyms.txt'))