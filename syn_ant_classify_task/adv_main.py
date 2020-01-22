import shutil
from os.path import join
from pathlib import Path

import numpy as np
from constants import THESAURUS_DIR
from dataset import Dataset
from model import AR, HRSWE
from syn_ant_classify_task.config import ori_thesauri, adv_thesauri, ths, ar_config, hrswe_config
from experiments import BaseExperiments, ARExperiments, HRSWEExperiments
from utils import generate_adv3, generate_sub_thesauri
from evaluate import SynAntClyEvaluator


# load datasets
dataset = Dataset()
dataset.load_task_datasets(*['adjective-pairs.val','noun-pairs.val','verb-pairs.val',
                            'adjective-pairs.test','noun-pairs.test','verb-pairs.test'])
dataset.load_words()
dataset.load_embeddings()


val_tasks = {name:task for name,task in dataset.tasks.items() if 'val' in name}
test_tasks = {name:task for name,task in dataset.tasks.items() if 'test' in name}
generate_sub_thesauri(join(THESAURUS_DIR, 'synonyms.txt'),ori_thesauri['syn_fname'],set(dataset.words))
generate_sub_thesauri(join(THESAURUS_DIR, 'antonyms.txt'),ori_thesauri['ant_fname'],set(dataset.words))

for r in [0.2,0.3,0.4,0.5]:
# for r in [0.1]:
    # prepare adv thesauri
    generate_adv3(r, ori_thesauri, dataset.tasks)
    # adv_thesauri['syn_fname'] = join(THESAURUS_DIR, 'clf', 'adv', '3', str(r), 'adv_synonyms.txt')
    # adv_thesauri['ant_fname'] = join(THESAURUS_DIR, 'clf', 'adv', '3', str(r), 'adv_antonyms.txt')
    dataset.load_thesauri(adv_thesauri)

    # evaluators
    hrswe_val = SynAntClyEvaluator(val_tasks, ths)
    hrswe_test = SynAntClyEvaluator(test_tasks, ths)
    ar_val = SynAntClyEvaluator(val_tasks, ths)
    ar_test = SynAntClyEvaluator(test_tasks,ths)

    # experiments
    ar_exp = ARExperiments(AR,ar_val,ar_test,dataset,ar_config)
    hrswe_exp = HRSWEExperiments(HRSWE,hrswe_val,hrswe_test,dataset,hrswe_config)

    # run exps
    ar_exp.run()

    hrswe_exp.run()

    # move files to dirs
    ta_dir = join('results','adv','3',str(r)) #3/3.1/3.2
    # create results dir
    Path(ta_dir).mkdir(parents=True, exist_ok=True)
    # move relevant files into results dir

    ar_fresults = ar_exp.config['exp_config']['exp_name'] + '_results' + '.pickle'
    ar_femb = ar_exp.config['exp_config']['exp_name'] + '_emb' + '.pickle'
    shutil.move(ar_fresults, join(ta_dir, ar_fresults))
    shutil.move(ar_femb, join(ta_dir, ar_femb))
    hrswe_fresults = hrswe_exp.config['exp_config']['exp_name'] + '_results' + '.pickle'
    hrswe_femb = hrswe_exp.config['exp_config']['exp_name'] + '_emb' + '.pickle'
    shutil.move(hrswe_fresults, join(ta_dir, hrswe_fresults))
    shutil.move(hrswe_femb, join(ta_dir, hrswe_femb))

    # create adv thesauri dir and move adv thesauri to that dir
    ta_dir = join(THESAURUS_DIR,'clf','adv','3',str(r)) #3/3.1/3.2
    Path(ta_dir).mkdir(parents=True, exist_ok=True)
    shutil.move(adv_thesauri['syn_fname'],join(ta_dir,'adv_synonyms.txt'))
    shutil.move(adv_thesauri['ant_fname'],join(ta_dir,'adv_antonyms.txt'))