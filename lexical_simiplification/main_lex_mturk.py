from pathlib import Path

from skopt import gp_minimize

from lexical_simiplification.helpers.experiments import LightLSExperiments

config = {
            'fdata': Path('task_data/lex_mturk_sen.txt'),
            'ftarget': Path('task_data/targets.pickle'),
            'fcandidates': Path('task_data/candidates.pickle'),
            'ftags': Path('task_data/pos_tags.pickle'),
            'foutdir': Path('task_data/simplified'),
            'fwordreqs': Path('task_data/unigram-freqs-en.txt'),
            'fembs': Path('../data/original_vecs/lexical_simplification.pickle'),
            'fstopwords': Path('task_data/stopwords-en.txt'),
            'minimizer': gp_minimize,
            'word_limit': None,
            'tholdcmplx': 0.0,
            'tholdsim': 0.0,
        }

exp = LightLSExperiments(config)
exp.run()