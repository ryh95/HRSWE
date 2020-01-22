from os.path import join

from constants import AR_THES_DIR
from word_sim_task.dataset import Dataset

thesauri = {'syn_fname': join(AR_THES_DIR, 'sub_synonyms.txt'),
                    'ant_fname': join(AR_THES_DIR, 'sub_antonyms.txt')}
# prepare dataset
dataset = Dataset(thesauri)
dataset.load_datasets()
dataset.generate_adv_thesaurus2(1.0)