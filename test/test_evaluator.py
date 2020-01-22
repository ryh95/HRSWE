import pickle
import time

from dataset import Dataset
from evaluate import SynAntClyEvaluator
import numpy as np

from syn_ant_classify_task.config import hrswe_config

with open('syn_ant_classify_test_val.pickle','rb') as f:
    emb_dict = pickle.load(f)


dataset = Dataset()
dataset.load_task_datasets(*['adjective-pairs.val','noun-pairs.val','verb-pairs.val',
                            'adjective-pairs.test','noun-pairs.test','verb-pairs.test'])
dataset.load_words()
dataset.load_embeddings()
val_tasks = {name:task for name,task in dataset.tasks.items() if 'val' in name}
ths = np.linspace(0,1,40)
val_evaluator = SynAntClyEvaluator(val_tasks, ths)
score,_ = val_evaluator.eval_emb_on_tasks(emb_dict)
val_evaluator.update_results()

test_evaluator = SynAntClyEvaluator(dataset.tasks,ths)
best_emb_dict = val_evaluator.best_emb
_, test_res = test_evaluator.eval_emb_on_tasks(best_emb_dict)

final_res = {
    'best_emb_dict':best_emb_dict,
    'test_res':test_res,
    'best_val_res':val_evaluator.best_results,
    'best_hyps':[0,1],
    'config':hrswe_config
}
if isinstance(test_evaluator,SynAntClyEvaluator):
    final_res['best_th'] = test_res['th']

with open(hrswe_config['exp_config']['exp_name']+'_results' + '.pickle', 'wb') as handle:
    pickle.dump(final_res, handle, protocol=pickle.HIGHEST_PROTOCOL)