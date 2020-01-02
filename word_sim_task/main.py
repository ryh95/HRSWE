from os.path import join

import numpy as np
from constants import THESAURUS_DIR
from dataset import Dataset
from model import AR, HRSWE, LHRSWE
from word_sim_task.config import ori_thesauri, ar_config, hrswe_config, lhrswe_config
from experiments import BaseExperiments, HRSWEExperiments
from utils import generate_sub_thesauri
from evaluate import WordSimEvaluator

# load datasets

dataset = Dataset()
dataset.load_task_datasets(*['SIMLEX999','SIMVERB3000-test','SIMVERB500-dev'])
dataset.load_words()
dataset.load_embeddings()


# prepare tasks to eval
val_tasks = {name:task for name,task in dataset.tasks.items() if 'val' in name or 'dev' in name}
test_tasks = {name:task for name,task in dataset.tasks.items() if name not in val_tasks}

# generate and load sub thesauri
generate_sub_thesauri(join(THESAURUS_DIR, 'synonyms.txt'),ori_thesauri['syn_fname'],set(dataset.words))
generate_sub_thesauri(join(THESAURUS_DIR, 'antonyms.txt'),ori_thesauri['ant_fname'],set(dataset.words))
dataset.load_thesauri(ori_thesauri)

# evaluators
hrswe_val = WordSimEvaluator(val_tasks)
hrswe_test = WordSimEvaluator(test_tasks)
ar_val = WordSimEvaluator(val_tasks)
ar_test = WordSimEvaluator(test_tasks)

# experiments
# ar_exp = BaseExperiments(AR,ar_val,ar_test,dataset,ar_config)
hrswe_exp = HRSWEExperiments(HRSWE,hrswe_val,hrswe_test,dataset,hrswe_config)

# run exps
hrswe_exp.run()

# ar_test_score = ar_exp.run()
# ar_test_fscore = ar_exp.config['exp_config']['exp_name'] + '_score' + '.npy'
# np.save(ar_test_fscore, ar_test_score)