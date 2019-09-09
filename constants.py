import os
from os.path import join

TOP_K = 25000
NUM_VOCAB = 20000
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = join(BASE_DIR,'data')

ORIGINAL_VECS_DIR = join(DATA_DIR, 'original_vecs')
ORIGINAL_EMBEDDING = join(ORIGINAL_VECS_DIR, 'GoogleNews-vectors-negative300.bin')

SPECIALIZED_VECS_DIR = join(DATA_DIR, 'specialized_vecs')

## processed vecs dir
SYN_ANT_CLASSIFY_VECS = join(SPECIALIZED_VECS_DIR, 'syn_ant_classify')
ATTRACT_REPEL_VECS = join(SPECIALIZED_VECS_DIR, 'attract_repel')

THESAURUS_DIR = join(DATA_DIR,'thesaurus')
WORD_NET_DIR = join(THESAURUS_DIR,'word_net')
ROGET_DIR = join(THESAURUS_DIR,'roget')
THESAURUS_COM_DIR = join(THESAURUS_DIR,'thesaurus_com')
WORD_ROGET_DIR = join(THESAURUS_DIR, 'AntonymPipeline')
AR_THES_DIR = join(THESAURUS_DIR,'attract_repel')


SIMLEX_DIR = join(DATA_DIR,'SimLex-999')
SIMVERB_DIR = join(DATA_DIR,'SimVerb-3500')
GRE_DIR = join(DATA_DIR,'GRE_antonym')

## task dir
SYN_ANT_CLASSIFY_TASK_DIR = join(BASE_DIR, 'syn_ant_classify_task')
WORD_SIM_TASK_DIR = join(BASE_DIR,'word_sim_task')
SENTIMENT_TASK_DIR = join(BASE_DIR,'sentiment_task')


VOCAB_FREQUENCY_DIR = join(DATA_DIR,'vocab_frequency')
VOCAB_FREQUENCY = join(VOCAB_FREQUENCY_DIR, 'lemmaed_count_1w.txt')
VOCAB_DIR = join(DATA_DIR,'vocab')
MATLAB_DIR = join(BASE_DIR,'matlab')


ATTRACT_REPEL_DIR = join(BASE_DIR,'attract_repel')