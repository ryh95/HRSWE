SGNS-GN.txt

reproduce: 

**experiments.py**
```python
method_name = ''
sel_vec_dir = join(ORIGINAL_VECS_DIR,method_name)
...
specialize = False
```
archived embedding:

original_vecs/syn_ant_classify_test_val.pickle

HRSWE

results:

best th: 0.102564
best beta: 54.555948
test adjective: 0.937531
test noun: 0.811682
test verb: 0.907368

test results obtained with our methods HRSWE, input is `original_vecs/syn_ant_classify_test_val.pickle`

reproduce:

**experiments.py**
```python
method_name = ''
sel_vec_dir = join(ORIGINAL_VECS_DIR,method_name)
...
specialize = True
...
config = {'sim_mat_type': 'pd', 'eig_vec_option': 'ld', 'emb_type': 0}
```
no archived embedding

LHRSWE

results:

best th: 0.076923
best beta: 54.555948
test adjective: 0.929032
test noun: 0.810152
test verb: 0.907368

test results obtained with our methods LHRSWE, input is `original_vecs/syn_ant_classify_test_val.pickle`

reproduce:

**experiments.py**
```python
method_name = ''
sel_vec_dir = join(ORIGINAL_VECS_DIR,method_name)
...
specialize = True
...
config = {'sim_mat_type': 'n', 'eig_vec_option': 'ld', 'emb_type': 1}
```
no archived embedding

Attract-Repel

results:

best f1 on dev: 2.758015, threshold: 0.256410

test adjective: 0.949633
test noun: 0.856031
test verb: 0.933194

test results obtained with Attract-Repel, input is data/specialized_vecs/attract_repel/paper_results/syn_ant_classify_test_val.pickle
(which is specialized by `original_vecs/syn_ant_classify_test_val.pickle`)

reproduce:

run **attract_repel.py** with the paper settings 

there is some randomness in the method, so the results would not be the same as those in the paper

archived embedding:

data/specialized_vecs/attract_repel/paper_results/syn_ant_classify_test_val.pickle

