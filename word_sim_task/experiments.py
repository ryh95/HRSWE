import os
import pickle
from collections import defaultdict
from os.path import join

from scipy import linalg
from six import iteritems
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils import Bunch
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_RW, fetch_RG65, fetch_MTurk, \
    fetch_TR9856
from web.embedding import Embedding
from web.evaluate import evaluate_similarity
import numpy as np
# import cvxpy as cp

from constants import SPECIALIZED_VECS_DIR, VOCAB_DIR, ORIGINAL_VECS_DIR, ORIGINAL_EMBEDDING, WORD_SIM_TASK_DIR, \
    AR_THES_DIR
from posdef import nearestPD,isPD

# load embedding_dict
from preprocess import StanfordSentimentProcesser, GeneralTextProcesser
from utils import prepare_syn_ant_graph, is_psd, \
    get_words_vecs_from_word_sim, combine_bunches, DAWE
from word_sim_task.local_utils import fetch_SimVerb3500, create_test_emb_on_word_sim_tasks, draw_results, \
    evaluate_similarity_on_tasks

if __name__ == '__main__':

    # with open('words_emb_n.pickle', 'rb') as handle:
    #     embedding_dict = pickle.load(handle)

    sim_tasks = {
            # "MEN": fetch_MEN(),
            # "WS353": fetch_WS353(),
            # "WS353S": fetch_WS353(which="similarity"),
            "SIMLEX999": fetch_SimLex999(),
            "SIMVERB3000-test" : fetch_SimVerb3500(join(WORD_SIM_TASK_DIR,'task_data','SimVerb-3000-test.txt')),
            "SIMVERB500-dev" : fetch_SimVerb3500(join(WORD_SIM_TASK_DIR,'task_data','SimVerb-500-dev.txt')),
            # "RW": fetch_RW()
            # "RG65": fetch_RG65(),
            # "MTurk": fetch_MTurk(),
            # "TR9856": fetch_TR9856(),
        }

    # test_on_sentiment = False

    # if test_on_sentiment == True:
    #     vocab_fname = '_'.join(sorted(sim_tasks.keys()))+'_sentiment'
    # else:
    #     vocab_fname = '_'.join(sorted(sim_tasks.keys()))

    vocab_fname = '_'.join(sorted(sim_tasks.keys()))
    # vocab_fname = 'SIMLEX999_SIMVERB3500'
    # vocab_fname = 'sentiment'

    if os.path.isfile(join(VOCAB_DIR,vocab_fname+'.npy')):
        words = np.load(join(VOCAB_DIR,vocab_fname+'.npy'))
    else:

        words = set()
        for name, data in iteritems(sim_tasks):
            words |= set(data.X.flatten())

        # words, word_emb = get_words_vecs_from_word_sim(sim_tasks, embedding_dict)
        # add sentiment words

        words = list(words)

        # if test_on_sentiment:
        #
        #     sentiment_processer = StanfordSentimentProcesser()
        #     sentiment_vocab = set()
        #     for fname in ['train.txt', 'dev.txt', 'test.txt']:
        #         sentiment_vocab |= sentiment_processer.extract_vocab_from_processed_file(
        #             join(SENTIMENT_DIR, 'processed_' + fname))
        #
        #     words = list(set(words) | sentiment_vocab)

        np.save(join(VOCAB_DIR,vocab_fname+'.npy'),words)

    # change to SPECIALIZED_VECS_DIR, set sepcialize to False and change method name
    # to see the result of specialized vectors
    # method_name = 'attract_repel/paper_results'
    method_name = ''
    sel_vec_dir = join(ORIGINAL_VECS_DIR,method_name)
    sel_vec_fname = join(sel_vec_dir,vocab_fname)
    # sel_vec_fname = 'paper_results/wn_ro_pd_ld_0'

    if os.path.isfile(sel_vec_fname+'.pickle'):
        with open(sel_vec_fname+'.pickle','rb') as handle:
            emb_dict = pickle.load(handle)

    else:
        text_preprocesser = GeneralTextProcesser()
        emb_dict = text_preprocesser.vocab2vec(words, sel_vec_dir, sel_vec_fname, ORIGINAL_EMBEDDING,
                                            ['pickle'], 'word2vec', normalize=True, oov_handle='mean_emb_vec')



    # benchmark
    # emb_obj = Embedding.from_dict({w: words_emb[:, i] for i, w in enumerate(words)})
    benchmark_scores = evaluate_similarity_on_tasks(sim_tasks, emb_dict)

    # sentiment benchmark
    # train_name = join(SENTIMENT_DIR, 'processed_train.txt')
    # dev_name = join(SENTIMENT_DIR, 'processed_dev.txt')
    # test_name = join(SENTIMENT_DIR, 'processed_test.txt')
    # gs = test_emb_on_sentiment(emb_obj, train_name, dev_name)
    # X_test, y_test = load_X_y(test_name, emb_obj)
    # print('test score on original embedding: %f' % gs.score(X_test,y_test))

    specialize = False
    thesauri_name = 'wn_ro'
    sim_type = 'n'
    eigen_vec_option = 'ld'
    emb_type = 1
    results_fname = '_'.join([thesauri_name, sim_type, eigen_vec_option, str(emb_type)])

    if specialize:

        # create and test embedding on word_sim tasks


        # betas = [0.5]
        beta1s = np.linspace(0, 1, 21)
        beta2s = np.linspace(0, 1, 21)
        # word_sim_pairs = combine_bunches(*sim_tasks.values())
        # thesauri = {'name':thesauri_name,'word_sim_pairs':word_sim_pairs}
        thesauri = {'syn_fname':join(AR_THES_DIR,'synonyms.txt'),'ant_fname':join(AR_THES_DIR,'antonyms.txt')}

        words_emb = [emb_dict[w] for w in words]
        words_emb = np.vstack(words_emb).T
        results = create_test_emb_on_word_sim_tasks(words, words_emb, {"SIMVERB500-dev":sim_tasks["SIMVERB500-dev"]}, beta1s,beta2s,
                                                    thesauri, sim_type, eigen_vec_option, emb_type)


        if not os.path.isfile(results_fname + '.pickle'):
            with open(results_fname + '.pickle', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


        evaluate_similarity_on_tasks(sim_tasks, results["best_scored_emb"])

        # use best emb on word sim task to test on sentiment data
        # gs = test_emb_on_sentiment(results['best_scored_emb'], train_name, dev_name)
        # X_test, y_test = load_X_y(test_name, results['best_scored_emb'])
        # print('test score on tunned embedding: %f' % gs.score(X_test,y_test))

    draw_word_sim = True
    if draw_word_sim:
        # results['benchmark_scores'] = benchmark_scores
        with open(results_fname + '.pickle', 'rb') as handle:
            results = pickle.load(handle)
        evaluate_similarity_on_tasks(sim_tasks, results["best_scored_emb"])
        draw_results(results, results_fname + '.html', ["SIMVERB500-dev"])