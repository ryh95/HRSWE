### basic setting

original_emb: word2vec (SGNS-GN)

vocab: simlex + simverb

lexical_constrain: atttract_repel/antonyms,synonyms (subset of the corresponding vocab)

### results

#### SGNS-GN

simlex: 0.442, simverb: 0.358

reproduce: 

**experiments.py**
```python
method_name = ''
sel_vec_dir = join(ORIGINAL_VECS_DIR,method_name)
...
specialize = False
```
archived embedding:

original_vecs/SIMLEX999_SIMVERB3000-test_SIMVERB500-dev.pickle

#### HRSWE

simlex: 0.693, simverb: 0.701

reproduce:

**experiments.py**
```python
method_name = ''
sel_vec_dir = join(ORIGINAL_VECS_DIR,method_name)
...
specialize = True
...
sim_type = 'pd'
emb_type = 0
```
archived embedding:

'wn_ro_pd_ld_0.pickle', best_scored_emb 

#### LHRSWE

simlex: 0.664, simverb: 0.631

reproduce:

**experiments.py**
```python
method_name = ''
sel_vec_dir = join(ORIGINAL_VECS_DIR,method_name)
...
specialize = True
...
sim_type = 'n'
emb_type = 1
```
archived embedding:

'wn_ro_n_ld_1.pickle', best_scored_emb
 
#### Attract-Repel

simlex: 0.780, simverb: 0.745

reproduce:

run **attract_repel.py** with the paper settings 

there is some randomness in the method, so the results would not be the same as those in the paper

archived embedding:

data/specialized_vecs/attract_repel/paper_results/SIMLEX999_SIMVERB3000-test_SIMVERB500-dev.pickle