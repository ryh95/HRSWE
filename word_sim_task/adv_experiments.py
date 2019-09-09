# select a subset of thesaurus
import pickle
from os.path import join

from six import iteritems
from web.datasets.similarity import fetch_SimLex999

from constants import VOCAB_DIR, WORD_SIM_TASK_DIR, AR_THES_DIR, ORIGINAL_VECS_DIR, ORIGINAL_EMBEDDING, \
    ATTRACT_REPEL_VECS
from preprocess import GeneralTextProcesser
from utils import select_syn_ant_sample, adv_attack_thesaurus
import numpy as np
import os

from word_sim_task.local_utils import fetch_SimVerb3500, evaluate_similarity_on_tasks, \
    create_test_emb_on_word_sim_tasks, draw_results

sim_tasks = {
            "SIMLEX999": fetch_SimLex999(),
            "SIMVERB3000-test" : fetch_SimVerb3500(join(WORD_SIM_TASK_DIR,'task_data','SimVerb-3000-test.txt')),
            "SIMVERB500-dev" : fetch_SimVerb3500(join(WORD_SIM_TASK_DIR,'task_data','SimVerb-500-dev.txt')),
        }

vocab_fname = '_'.join(sorted(sim_tasks.keys()))


if os.path.isfile(join(VOCAB_DIR, vocab_fname + '.npy')):
    words = np.load(join(VOCAB_DIR, vocab_fname + '.npy'))
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

    np.save(join(VOCAB_DIR, vocab_fname + '.npy'), words)

# select subset
sub_syn_fname = join(AR_THES_DIR,'sub_synonyms.txt')
sub_ant_fname = join(AR_THES_DIR,'sub_antonyms.txt')
if not os.path.isfile(sub_syn_fname) and not os.path.isfile(sub_ant_fname):
    select_syn_ant_sample(join(AR_THES_DIR,'synonyms.txt'),sub_syn_fname,words)
    select_syn_ant_sample(join(AR_THES_DIR,'antonyms.txt'),sub_ant_fname,words)

# generate adversarial subset
adv_sub_syn_fname = join(AR_THES_DIR,'adv_sub_synonyms.txt')
adv_sub_ant_fname = join(AR_THES_DIR,'adv_sub_antonyms.txt')
if not os.path.isfile(adv_sub_syn_fname) and not os.path.isfile(adv_sub_ant_fname):
    adv_attack_thesaurus(sub_syn_fname,sub_ant_fname)


# change to SPECIALIZED_VECS_DIR, set sepcialize to False and change method name
# to see the result of specialized vectors
# method_name = 'attract_repel/paper_results'
method_name = ''
sel_vec_dir = join(ATTRACT_REPEL_VECS,method_name)
# sel_vec_dir = join(ORIGINAL_VECS_DIR,method_name)
sel_vec_fname = join(sel_vec_dir,'adv_SIMLEX999_SIMVERB3000-test_SIMVERB500-dev')
# sel_vec_fname = join(sel_vec_dir,vocab_fname)
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
is_adv = 1
results_fname = '_'.join([thesauri_name, sim_type, eigen_vec_option, str(emb_type),str(is_adv)])

if specialize:

    # create and test embedding on word_sim tasks


    # betas = [0.5]
    beta1s = np.linspace(0, 1, 21)
    beta2s = np.linspace(0, 1, 21)
    # word_sim_pairs = combine_bunches(*sim_tasks.values())
    # thesauri = {'name':thesauri_name,'word_sim_pairs':word_sim_pairs}
    thesauri = {'syn_fname':join(AR_THES_DIR,'adv_sub_synonyms.txt'),'ant_fname':join(AR_THES_DIR,'adv_sub_antonyms.txt')}

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

draw_word_sim = False
if draw_word_sim:
    # results['benchmark_scores'] = benchmark_scores
    with open(results_fname + '.pickle', 'rb') as handle:
        results = pickle.load(handle)
    evaluate_similarity_on_tasks(sim_tasks, results["best_scored_emb"])
    draw_results(results, results_fname + '.html', ["SIMVERB500-dev"])