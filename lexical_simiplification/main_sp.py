import pickle
from os.path import join

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler

from constants import THESAURUS_DIR, VOCAB_DIR, ORIGINAL_VECS_DIR
from dataset import Dataset
from model import AR, HRSWE, generate_syn_ant_graph

dataset = Dataset()
# dataset.load_task_datasets(*['adjective-pairs.val','noun-pairs.val','verb-pairs.val',
#                             'adjective-pairs.test','noun-pairs.test','verb-pairs.test'])
dataset.vocab_fname = 'lexical_simplification'
dataset.load_words()
dataset.load_embeddings()
ori_thesauri = {'syn_fname': join(THESAURUS_DIR, 'lexical_simplification', 'synonyms.txt'),
                'ant_fname': join(THESAURUS_DIR, 'lexical_simplification', 'antonyms.txt')}
dataset.load_thesauri(ori_thesauri)

# syn_mar, ant_mar, batch_size, epoch_num, l2_reg = 0.7,0.4,128,10,10**-6
# model = AR(syn_mar, ant_mar, batch_size, epoch_num, l2_reg)
# sp_emb = model.specialize_emb(dataset.emb_dict,
#                               dataset.syn_pairs,dataset.ant_pairs)
# sp_emb_dict = {w:sp_emb[i,:] for i,w in enumerate(dataset.words)}
# with open('lexical_simplification_sp.pickle','wb') as f:
#     pickle.dump(sp_emb_dict,f,pickle.HIGHEST_PROTOCOL)
beta0, beta1, beta2, mis_syn, pos, neg = 0.5,0.7,0.6,0,0.4,0.3
emb = [vec for vec in dataset.emb_dict.values()]
emb = np.vstack(emb).astype(np.float32).T
# scaler = StandardScaler()
# emb = scaler.fit_transform(emb.T).T
d,n = emb.shape
W = emb.T @ emb
adj_pos,adj_neg,G = generate_syn_ant_graph(dataset.words,dataset.syn_pairs,dataset.ant_pairs)
adj_spread = nx.adjacency_matrix(G, nodelist=dataset.words)

model = HRSWE(beta0, beta1, beta2, mis_syn, pos, neg,adj_pos=adj_pos,adj_neg=adj_neg,adj_spread=adj_spread,W=W,n=n,d=d)
sp_emb = model.specialize_emb(dataset.emb_dict,
                              dataset.syn_pairs,dataset.ant_pairs)
sp_emb_dict = {w:sp_emb[i,:] for i,w in enumerate(dataset.words)}
with open('lexical_simplification.pickle','wb') as f:
    pickle.dump(sp_emb_dict,f,pickle.HIGHEST_PROTOCOL)
# merge sp and non sp
# words = np.load(join(VOCAB_DIR, 'lexical_simplification' + '.npy'))
# with open(join(ORIGINAL_VECS_DIR,'lexical_simplification')+'.pickle','rb') as f:
#     emb_dict = pickle.load(f)
# merge_sp_emb_dict = {}
# for w in words:
#     if w in sp_emb_dict:
#         merge_sp_emb_dict[w] = sp_emb_dict[w]
#     else:
#         merge_sp_emb_dict[w] = emb_dict[w]
# with open('lexical_simplification.pickle','wb') as f:
#     pickle.dump(merge_sp_emb_dict,f,pickle.HIGHEST_PROTOCOL)