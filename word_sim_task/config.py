import numpy as np
from os.path import join

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from constants import AR_THES_DIR, THESAURUS_DIR

# public config
adv_thesauri = {'syn_fname': join(THESAURUS_DIR, 'sim', 'adv_synonyms.txt'),
                'ant_fname': join(THESAURUS_DIR, 'sim', 'adv_antonyms.txt')}
ori_thesauri = {'syn_fname': join(THESAURUS_DIR, 'sim', 'synonyms.txt'),
                'ant_fname': join(THESAURUS_DIR, 'sim', 'antonyms.txt')}
public_hyp_config = {
    'n_calls':60,
    'verbose':True
}

# HRSWE config

hrswe_exp_config = {
    'save_res':True,
    'exp_name':'hrswe'
}

hrswe_config = {
    'exp_config':hrswe_exp_config,
    'hyp_tune_func':gp_minimize,
    'hyp_opt_space':[
        Categorical([1]),# beta0
        # Real(10**-3,10**2,'log-uniform'),
        # Real(10**-3,10**2,'log-uniform'),
        Real(0,1),# beta1
        Real(0,1),# beta2
        # Real(0,1), # W_max
        # Real(-1,0) # W_min
        # Real(10**-3,10**2,'log-uniform'),# beta3
        # Real(10**-3,10**2,'log-uniform'),# beta4
    ],
    'tune_func_config': public_hyp_config,

}


# LHRSWE
lhrswe_exp_config = {
    'save_res':True,
    'exp_name':'lhrswe'
}

lhrswe_config = {
    'exp_config':lhrswe_exp_config,
    'hyp_tune_func':gp_minimize,
    'hyp_opt_space':[
        Categorical([1]),# beta0
        Real(0,1),# beta1
        Real(0,1),# beta2
        # Real(0,1), # W_max
        # Real(-1,0) # W_min
        # Real(10**-3,10**2,'log-uniform'),# beta3
        # Real(10**-3,10**2,'log-uniform'),# beta4
    ],
    'tune_func_config': public_hyp_config,

}


# Retrofitted matrix
matrix_exp_config = {
    'save_res':True,
    'exp_name':'lhrswe'
}

matrix_config = {
    'exp_config':matrix_exp_config,
    'hyp_tune_func':gp_minimize,
    'hyp_opt_space':[
        Categorical([1]),# beta0
        Real(0,1),# beta1
        Real(0,1),# beta2
        Categorical([0.4]),
        Categorical([0.6])
        # Real(10**-3,10**2,'log-uniform'),# beta3
        # Real(10**-3,10**2,'log-uniform'),# beta4
    ],
    'tune_func_config': public_hyp_config,

}

# AR config

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
