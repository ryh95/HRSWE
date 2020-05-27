from math import inf

from lexical_simiplification.helpers import lightls
from utils import blockPrinting


class LightLSEvaluator(object):

    def __init__(self,wfs,stopwords):
        self.cur_score = None
        self.best_score = -inf
        self.cur_emb = None
        self.best_emb = None
        self.cur_parameters = None
        self.best_parameters = None
        self.wfs = wfs
        self.stopwords = stopwords
        # self.eval_data =
        # self.eval_targets =
        # self.eval_pos_tags =
        # self.eval_candidates =

    @blockPrinting
    def update_results(self):
        if self.cur_score > self.best_score:
            self.best_score = self.cur_score
            self.best_emb = self.cur_emb
            self.best_parameters = self.cur_parameters
            print('Current best eval score: %f' % (self.best_score))

    def evaluate_emb(self,embeddings,parameters):
        simplifier = lightls.LightLS(embeddings, self.wfs, parameters, self.stopwords)
        simplifications = simplifier.simplify_lex_mturk(self.eval_data, self.eval_targets, self.eval_pos_tags)
        self.cur_score, change = simplifier.evaluate_lex_mturk_simplification(simplifications, self.eval_candidates)
        self.cur_emb = embeddings
        self.cur_parameters = parameters
        return self.cur_score