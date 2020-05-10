import pickle
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from skopt import dump
from skopt.space import Integer, Real

from lexical_simiplification.helpers import text_embeddings
from lexical_simiplification.helpers import io_helper
from lexical_simiplification.helpers import lightls


class LightLSExperiments(object):

    def __init__(self,config):

        self.config = config
        # preparation

        print("Loading unigram frequencies...")
        ls = io_helper.load_lines(config['fwordreqs'])
        self.wfs = {x.split()[0].strip(): int(x.split()[1].strip()) for x in ls}

        if config['fembs'].suffix == '.txt':
            self.embeddings = text_embeddings.Embeddings()
            self.embeddings.load_embeddings(config['fembs'], config['word_limit'], language='default', print_loading=True, skip_first_line=False,
                                         normalize=True)
            self.embeddings.inverse_vocabularies()
        elif config['fembs'].suffix == '.bin':
            self.embeddings = text_embeddings.Word2VecEmbedding()
            self.embeddings.load_embeddings(config['fembs'], limit=None, language='default', print_loading=True,
                                         skip_first_line=False, normalize=True)
        elif config['fembs'].suffix == '.pickle':
            self.embeddings = text_embeddings.PickleEmbedding()
            self.embeddings.load_embeddings(config['fembs'], config['word_limit'], language='default',
                                            print_loading=True, skip_first_line=False,
                                            normalize=True)
            # self.embeddings.inverse_vocabularies()


        self.stopwords = io_helper.load_lines(config['fstopwords']) if config['fstopwords'] else None

        with open(config['ftarget'], 'rb') as f_targets:
            self.targets = pickle.load(f_targets)
        with open(config['fcandidates'], 'rb') as f_candidates:
            self.candidates = pickle.load(f_candidates)

    def minimize(self, space, **min_args):
        minimizer = self.config['minimizer']
        return minimizer(self.objective, space, **min_args)

    def split_lex_mturk(self,test_size=0.4):
        """
        fdata: Path object
        """
        lines = self.config['fdata'].read_text()
        sens = [line.split() for line in lines.split('\n')[:-1]]
        X_val, X_test, val_targets, test_targets, val_candidates, test_candidates = \
            train_test_split(sens,self.targets,self.candidates, test_size=test_size)
        Xs = [[X_val,X_test],[val_targets,test_targets],[val_candidates,test_candidates]]
        fs,types = ['fdata','ftarget','fcandidates'],['val','test']
        for i,f in enumerate(fs):
            fstem = self.config[f].stem
            for j,type in enumerate(types):
                f_type = self.config[f].parent / (fstem + f'_{type}' + '.pickle')
                with open(f_type,'wb') as handle:
                    pickle.dump(Xs[i][j],handle,pickle.HIGHEST_PROTOCOL)

        return X_val, X_test, val_targets, test_targets, val_candidates, test_candidates

    def objective(self,feasible_point):
        parameters = {"complexity_drop_threshold": feasible_point[2], "num_cand": feasible_point[0],
                      "similarity_threshold": self.config['tholdsim'], "context_window_size": feasible_point[1],
                      "complexity_threshold": self.config['tholdcmplx']}
        simplifier = lightls.LightLS(self.embeddings, self.wfs, parameters, self.stopwords)
        simplifications = simplifier.simplify_lex_mturk(self.eval_data, self.eval_targets)
        acc, change = simplifier.evaluate_lex_mturk_simplification(simplifications, self.eval_candidates)
        return -acc

    def run(self):

        # split text
        fstem = self.config['fdata'].stem
        fval = self.config['fdata'].parent / (fstem + '_val' + '.pickle')
        if not Path(fval).exists():
            X_val, X_test, val_targets, test_targets, val_candidates, test_candidates = self.split_lex_mturk()
            Xs = [[X_val, X_test], [val_targets, test_targets], [val_candidates, test_candidates]]
        else:
            Xs = []
            fs, types = ['fdata', 'ftarget', 'fcandidates'], ['val', 'test']
            for i, f in enumerate(fs):
                fstem = self.config[f].stem
                stem = []
                for j, type in enumerate(types):
                    f_type = self.config[f].parent / (fstem + f'_{type}' + '.pickle')
                    with open(f_type, 'rb') as handle:
                        stem.append(pickle.load(handle))
                Xs.append(stem)

        # optimize on the val data and test on the testing data
        self.eval_data, self.eval_targets, self.eval_candidates = [x[0] for x in Xs]
        x0 = [2,5,0.03]
        space = [
            Integer(2, 300),
            Integer(2, 10),
            Real(10 ** -5, 10 ** -1,'log-uniform')
        ]
        res = self.minimize(space, x0=x0, n_calls=50, verbose=True)

        self.eval_data, self.eval_targets, self.eval_candidates = [x[1] for x in Xs]
        print(f'test acc: {-self.objective(res.x)}')
        dump(res,'res-hyp.pickle',store_objective=False)