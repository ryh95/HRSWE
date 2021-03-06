import pickle
from os.path import join
from pathlib import Path

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
import numpy as np
from constants import THESAURUS_DIR, VOCAB_DIR, ORIGINAL_VECS_DIR
from dataset import Dataset
from lexical_simiplification.helpers import io_helper
from lexical_simiplification.helpers.evaluate import LightLSEvaluator
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

config = {
            # 'exp_name': exp_name,
            'fdata': Path('task_data/lex_mturk_sen.pickle'),
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
                # Categorical([10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1])
            ],
            'ls_opt_x0': [10, 5, 0.03],
            # 'ls_opt_x0': [10, 5, 10 ** -4],
            'ls_n_calls': 40,
            # 'ls_fixed_paras': [20,5,0.01],
            'word_limit': None,
            'tholdcmplx': 0.0,
            'tholdsim': 0.0,
        }

print("Loading unigram frequencies...")
ls = io_helper.load_lines(config['fwordreqs'])
wfs = {x.split()[0].strip(): int(x.split()[1].strip()) for x in ls}
stopwords = io_helper.load_lines(config['fstopwords']) if config['fstopwords'] else None
for i in range(3):
    data_dir = Path('task_data') / str(i)
    data_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(str(i))
    res_dir.mkdir(parents=True, exist_ok=True)
    # config['exp_id'] = i # create id of the val, test data and exp
    for j in range(3):
        for exp_name in ['hrswe','ar']:
            config['exp_name'] = exp_name
            # exp_res_dir = Path(exp_name+'_'+str(j)+'_'+'2') # hrswe-2
            exp_res_dir = Path(exp_name+'_'+str(j))
            exp_res_dir.mkdir(parents=True, exist_ok=True)

            if exp_name == 'hrswe':
                sp_model = HRSWE
                sp_opt_space = [
                    Real(0, 1),
                    # Real(10 ** -1, 10 ** 1, 'log-uniform'),
                    # Categorical([1]),
                    Real(10 ** -1, 10 ** 1, 'log-uniform'),
                    Real(10 ** -1, 10 ** 1, 'log-uniform'),
                    # Real(0, 1),
                    # Real(0, 1),
                    # Categorical([1]),
                    Categorical([0]),
                    Real(0, 1),  # W_max
                    Real(0, 1),  # W_min
                    # Categorical([0]),
                    # Categorical([0]),
                ]
                # sp_opt_x0 = [0.5, 0.7, 0.6, 0, 0.4, 0.3]
                # sp_opt_x0 = [0.7,6,4,0,0.4,0.4]
                sp_opt_x0 = None
                sp_n_calls = 30
            elif exp_name == 'ar':
                sp_model = AR
                sp_opt_space = [
                    Real(0, 1),  # syn mar
                    Real(0, 1),  # ant mar
                    Categorical([32, 64, 128, 256, 512, 1024]),  # batch size
                    Integer(1, 15),  # epoch num
                    Real(10 ** -9, 10 ** 0, 'log-uniform'),  # l2 reg
                ]
                # sp_opt_x0 = [0.7, 0.4, 128, 10, 10 ** -6]
                sp_opt_x0 = None
                sp_n_calls = 30

            config['sp_opt_space'] = sp_opt_space
            config['sp_opt_x0'] = sp_opt_x0
            config['sp_n_calls'] = sp_n_calls

            evaluator = LightLSEvaluator(wfs, stopwords)
            exp = SpLightLSExperiments(sp_model,dataset,config,evaluator)
            exp.run()

            Path('res-hyp.pickle').rename(exp_res_dir / 'res-hyp.pickle')
            Path('sp_time.pickle').rename(exp_res_dir / 'sp_time.pickle')
            Path('test_acc.pickle').rename(exp_res_dir / 'test_acc.pickle')
            Path('eval_hyp.pickle').rename(exp_res_dir / 'eval_hyp.pickle')

    for k in range(3):
        for exp_name in ['hrswe', 'ar']:
            # exp_res_dir = Path(exp_name + '_' + str(k)+'_'+'2') # hrswe-2
            exp_res_dir = Path(exp_name + '_' + str(k))
            exp_res_dir.rename(res_dir / exp_res_dir)

    # move test and val data
    fs, types = ['fdata', 'ftarget', 'fcandidates', 'ftags'], ['val', 'test']
    for f in fs:
        fstem = config[f].stem
        stem = []
        for type in types:
            f_type = config[f].parent / (fstem + f'_{type}' + '.pickle')
            f_type.rename(data_dir / (fstem + f'_{type}' + '.pickle'))