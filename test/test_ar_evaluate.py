import pickle
from math import inf
from os.path import join

from numpy import dot
from numpy.linalg import norm
from scipy.stats import spearmanr
from web.datasets.similarity import fetch_SimLex999

from constants import ORIGINAL_VECS_DIR, WORD_SIM_TASK_DIR
from word_sim_task.dataset import Dataset
from word_sim_task.evaluate import evaluator


def ar_evaluate(word_vectors,eval_data):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors.
    """
    pair_list = []


    for word_pair,score in zip(eval_data.X,eval_data.y):
        pair_list.append((tuple(word_pair),score))

    pair_list.sort(key=lambda x: - x[1])

    coverage = len(pair_list)

    extracted_list = []
    extracted_scores = {}

    for (x, y) in pair_list:
        (word_i, word_j) = x
        current_distance = distance(word_vectors[word_i], word_vectors[word_j])
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)
    return spearman_rho[0], coverage

def distance(v1, v2, normalised_vectors=False):
    """
    Returns the cosine distance between two vectors.
    If the vectors are normalised, there is no need for the denominator, which is always one.
    """
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / (norm(v1) * norm(v2))

if __name__ == '__main__':
    # eval_data = fetch_SimLex999()
    eval_data = Dataset().load_SimVerb3500(join(WORD_SIM_TASK_DIR, 'task_data', 'SimVerb-3000-test.txt'))
    emb_fname = join(ORIGINAL_VECS_DIR, 'SIMLEX999_SIMVERB3000-test_SIMVERB500-dev')
    evaluator_obj = evaluator(best_eval_score=-inf, tasks={"SIMVERB500-dev": None})
    # IMPORTANT: two spearmanr results are a little different
    with open(emb_fname + '.pickle', 'rb') as handle:
        emb_dict = pickle.load(handle)

    print(ar_evaluate(emb_dict,eval_data))
    evaluator_obj.eval_emb_on_sim({'test':eval_data},emb_dict)
