import pickle
from os.path import join
from pathlib import Path

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
import numpy as np
from constants import THESAURUS_DIR, VOCAB_DIR, ORIGINAL_VECS_DIR
from dataset import Dataset
from lexical_simiplification.helpers.experiments import LightLSExperiments, SpLightLSExperiments
from model import HRSWE, AR

dataset = Dataset()
dataset.vocab_fname = 'lexical_simplification_constrain'
dataset.load_words()
dataset.load_embeddings()
ori_thesauri = {'syn_fname': join(THESAURUS_DIR, 'lexical_simplification', 'synonyms.txt'),
                'ant_fname': join(THESAURUS_DIR, 'lexical_simplification', 'antonyms.txt')}
dataset.load_thesauri(ori_thesauri)
words = np.load(join(VOCAB_DIR, 'lexical_simplification' + '.npy'))
with open(join(ORIGINAL_VECS_DIR,'lexical_simplification')+'.pickle','rb') as f:
    emb_dict = pickle.load(f)
dataset.task_vocab = words
dataset.task_emb_dict = emb_dict
for i in range(5):
    for exp_name in ['hrswe']:
        exp_res_dir = Path(exp_name+'_'+str(i))
        exp_res_dir.mkdir(parents=True, exist_ok=True)

        if exp_name == 'hrswe':
            sp_model = HRSWE
            sp_opt_space = [
                Real(0, 1),
                # Categorical([1]),
                Real(0, 1),
                Real(0, 1),
                # Categorical([1]),
                Categorical([0]),
                # Real(0, 1),  # W_max
                # Real(0, 1),  # W_min
                Categorical([0]),
                Categorical([0]),
            ]
            # sp_opt_x0 = [0.5, 0.7, 0.6, 0, 0.4, 0.3]
            sp_opt_x0 = [0.5,0.7,0.6,0,0,0]
            sp_n_calls = 25
        elif exp_name == 'ar':
            sp_model = AR
            sp_opt_space = [
                Real(0, 1),  # syn mar
                Real(0, 1),  # ant mar
                Categorical([32, 64, 128, 256, 512, 1024]),  # batch size
                Integer(1, 15),  # epoch num
                Real(10 ** -9, 10 ** 0, 'log-uniform'),  # l2 reg
            ]
            sp_opt_x0 = [0.7, 0.4, 128, 10, 10 ** -6]
            sp_n_calls = 25

        config = {
            'exp_name': exp_name,
            'fdata': Path('task_data/lex_mturk_sen.txt'),
            'ftarget': Path('task_data/targets.pickle'),
            'fcandidates': Path('task_data/candidates.pickle'),
            'ftags': Path('task_data/pos_tags.pickle'),
            'foutdir': Path('task_data/simplified'),
            'fwordreqs': Path('task_data/unigram-freqs-en.txt'),
            'fembs': Path('../data/original_vecs/glove.42B.300d.txt'),
            'fstopwords': Path('task_data/stopwords-en.txt'),
            'minimizer': gp_minimize,
            'ls_opt_space': [
                Integer(2, 50),
                Integer(2, 10),
                Real(10 ** -6, 10 ** -1, 'log-uniform')
            ],
            'ls_opt_x0': [10, 5, 0.03],
            'ls_n_calls': 80,
            'sp_opt_space': sp_opt_space,
            'sp_opt_x0': sp_opt_x0,
            'sp_n_calls': sp_n_calls,
            'word_limit': None,
            'tholdcmplx': 0.0,
            'tholdsim': 0.0,
        }

        exp = SpLightLSExperiments(sp_model,dataset,config)
        exp.run()

        Path('res-hyp.pickle').rename(exp_res_dir / 'res-hyp.pickle')
        Path('sp_time.pickle').rename(exp_res_dir / 'sp_time.pickle')
        Path('test_acc.pickle').rename(exp_res_dir / 'test_acc.pickle')