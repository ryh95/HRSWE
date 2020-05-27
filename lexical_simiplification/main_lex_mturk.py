from pathlib import Path

from skopt import gp_minimize

from lexical_simiplification.helpers import io_helper
from lexical_simiplification.helpers.evaluate import LightLSEvaluator
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
print("Loading unigram frequencies...")
ls = io_helper.load_lines(config['fwordreqs'])
wfs = {x.split()[0].strip(): int(x.split()[1].strip()) for x in ls}
stopwords = io_helper.load_lines(config['fstopwords']) if config['fstopwords'] else None
evaluator = LightLSEvaluator(wfs,stopwords)
exp = LightLSExperiments(config,evaluator)
exp.run()