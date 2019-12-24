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
ths = np.linspace(0,1,40)
public_hyp_config = {
    'tune_func':gp_minimize,
    'n_calls':50,
    'verbose':True
}

# HRSWE config
# model_config(HRSWE config),hyp_tune_config(opt_space, bayes func),exp_config(save_emb,exp_name)

hrswe_model_config = {
    'sim_mat_type':'pd',
    'eig_vec_option':'ld',
    'emb_type':0
}
hrswe_hyp_config = {
    'opt_space':[
        Real(10**-3,10**2,'log-uniform'),# beta1
        Real(10**-3,10**2,'log-uniform'),# beta2
    ],
    'x0':[0.8,0.8]
}
hrswe_hyp_config = {**public_hyp_config,**hrswe_hyp_config}

hrswe_exp_config = {
    'save_res':True,
    'exp_name':'hrswe'
}

hrswe_config = {
    'model_config':hrswe_model_config,
    'hyp_tune_config': hrswe_hyp_config,
    'exp_config':hrswe_exp_config
}


# AR config
ar_model_config = {}
ar_hyp_config = {
    'opt_space':[
        Real(0,1), # syn mar
        Real(0,1), # ant mar
        Categorical([64]), # batch size
        Integer(1,20), # epoch num
        Real(10**-9,10**0,'log-uniform'), # l2 reg
    ],
    'x0':[0.4,0.6,64,10,10**-8]
}
ar_hyp_config = {**public_hyp_config,**ar_hyp_config}
ar_exp_config = {
    'save_res':True,
    'exp_name':'ar'
}

ar_config = {
    'model_config':ar_model_config,
    'hyp_tune_config': ar_hyp_config,
    'exp_config':ar_exp_config
}
