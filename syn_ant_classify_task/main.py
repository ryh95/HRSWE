import shutil
from os.path import join
from pathlib import Path

import numpy as np
from constants import THESAURUS_DIR
from dataset import Dataset
from experiments import BaseExperiments, ARExperiments, HRSWEExperiments
from model import AR, HRSWE
from syn_ant_classify_task.config import ori_thesauri, adv_thesauri, ths, ar_config, hrswe_config

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

dataset.load_thesauri(ori_thesauri)
# evaluators
hrswe_val = SynAntClyEvaluator(val_tasks, ths)
hrswe_test = SynAntClyEvaluator(test_tasks, ths)
ar_val = SynAntClyEvaluator(val_tasks, ths)
ar_test = SynAntClyEvaluator(test_tasks,ths)

# experiments

hrswe_exp = HRSWEExperiments(HRSWE,hrswe_val,hrswe_test,dataset,hrswe_config)
hrswe_exp.run()

ar_exp = ARExperiments(AR,ar_val,ar_test,dataset,ar_config)
ar_exp.run()