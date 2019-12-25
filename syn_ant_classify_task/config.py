import numpy as np
from os.path import join

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from constants import AR_THES_DIR, THESAURUS_DIR

# public config
adv_thesauri = {'syn_fname': join(THESAURUS_DIR, 'clf', 'adv_synonyms.txt'),
                'ant_fname': join(THESAURUS_DIR, 'clf', 'adv_antonyms.txt')}
ori_thesauri = {'syn_fname': join(THESAURUS_DIR, 'clf', 'synonyms.txt'),
                'ant_fname': join(THESAURUS_DIR, 'clf', 'antonyms.txt')}
ths = np.linspace(-1,1,40)
public_hyp_config = {
    'n_calls':20,
    'verbose':True
}

# HRSWE config
# model_config(HRSWE config),hyp_tune_config(opt_space, bayes func),exp_config(save_emb,exp_name)

# hrswe_hyp_config = {
#     'x0':[0.8,0.8]
# }
# hrswe_hyp_config = {**public_hyp_config,**hrswe_hyp_config}

hrswe_exp_config = {
    'save_res':True,
    'exp_name':'hrswe'
}

hrswe_config = {
    'exp_config':hrswe_exp_config,
    'hyp_tune_func':gp_minimize,
    'hyp_opt_space':[
        Real(10**-3,10**2,'log-uniform'),# beta1
        Real(10**-3,10**2,'log-uniform'),# beta2
    ],
    'tune_func_config': public_hyp_config,

}


# AR config
# ar_hyp_config = {
#     'x0':[0.4,0.6,64,10,10**-8]
# }
# ar_hyp_config = {**public_hyp_config,**ar_hyp_config}
ar_exp_config = {
    'save_res':True,
    'exp_name':'ar'
}

ar_config = {
    'exp_config':ar_exp_config,
    'hyp_tune_func':gp_minimize,
    'hyp_opt_space':[
        Real(0,1), # syn mar
        Real(0,1), # ant mar
        Categorical([64]), # batch size
        Integer(1,20), # epoch num
        Real(10**-9,10**0,'log-uniform'), # l2 reg
    ],
    'tune_func_config': public_hyp_config,
}
