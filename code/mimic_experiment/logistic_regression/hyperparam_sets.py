import sys, itertools
from pathlib import Path
# add our ./code/logistic_regression_experiment/ folder so we can find our config file 
sys.path.append(Path(".", "code", "mimic_experiment").as_posix())
sys.path.append(Path(".", "code").as_posix())
from config import *
from data_utils import *
from experiment_utils import *
from numpy import arange

final_n = 10 # runs to do for final test


## hyperparam utils: 
def _get_h_from_dict(h_key, hyperparameter_set, fixed):
    """used by functions below to get list of hyperparm dicts. 
    Args:
        h_key (tuple): key for hyperparmeter_set, (target, h_pass, loss) or (target, h_pass, loss, epsilon)..
        hyperparameter_set (dict): defined below in this script 
        fixed (dict): fixed hyperparameter names: values
    Returns:
        list of hyperparm dicts. 
    """
    N, LR_dist = hyperparameter_set[h_key]
    if type(LR_dist) == DictDist:
        LR_hyperparams_list = LR_dist.rvs(N)
    else:
        # get seeds first
        LR_hyperparams_list = DictDist({'seed': ss.randint(1, 10000),}).rvs(N)
        for h in LR_hyperparams_list:
            h.update(LR_dist)
            
    model_id = 0 # start at 0 and count up 
    for h in LR_hyperparams_list: # add in the fixed hyperparams
        h.update(fixed)
        model_id += 1
        h.update({'model_id': model_id})
    return LR_hyperparams_list


def get_h_LR_nonpriv(target,  h_pass, loss, run, hyperparameter_set, use_full_train, 
                     use_gpu, folder=LR_BASELINE_RESULTS_FOLDER, final_n = final_n): 
    """called in the run_nonprivate.py and the timing_benchmark_nonprivate.py scripts. 
    This returns the list of hyperparmeters for the kernel function
    """
    if h_pass in [ "final", "benchmarks"]: 
        refined_path = Path(folder, f'refined{run}')
        assert refined_path.exists(), f"{refined_path} does not exist!\n h_pass = 'final' and run = {run} requires {refined_path.as_posix()} to exist. (this is created when running run_nonprivate.py with h_pass = 'refined'.)" 
        nonpriv_df = pd.DataFrame.from_records( map( unjsonify,refined_path.rglob('*.json') ))
        for p in itertools.product(['mort_icu', 'los_3'], ['final'], ['l2', 'bce']):
            df = nonpriv_df[(nonpriv_df.target == p[0]) & (nonpriv_df.loss == p[2])]
            best = df.nlargest(1, 'auc') 
            clean = best.drop(labels = ['accuracy', 'aps', 'auc', 'dist_to_[0,1]', 'f1', 'fpr',
                   'loss', 'model_id', 'prec', 'seed', 'target', 'thresh', 'tpr',
                   'use_full_train', 'used_gpu'], axis = 1).to_dict('records')[0]        
            hyperparameter_set[p] = (final_n, clean)
            hyperparameter_set[(p[0], 'benchmarks', p[2])] = (1, non_priv_hyperparameter_set[p][1])
        
    fixed = {"target": target,
            "use_full_train": use_full_train,
            "loss": loss,
            "loss_reduction": 'mean',
            "used_gpu" : use_gpu}

    return _get_h_from_dict((target, h_pass, loss), hyperparameter_set, fixed)


def get_h_LR_dpsgd(target, h_pass, loss, epsilons, run, hyperparameter_set, use_full_train, 
                  use_gpu, folder=LR_DPSGD_RESULTS_FOLDER, final_n = final_n): 
    """called in the run_dpsgd.py and the timing_benchmark_nonprivate.py scripts. 
    This returns the list of hyperparmeters for the kernel function
    """
    if h_pass in [ "final", "benchmarks"]: 
        refined_path = Path(folder, f'refined{run}')
        assert refined_path.exists(), f"{refined_path} does not exist!\n h_pass = 'final' and run = {run} requires {refined_path.as_posix()} to exist. (this is created when running run_nonprivate.py with h_pass = 'refined'.)" 
        dpsgd_df = pd.DataFrame.from_records( map( unjsonify,refined_path.rglob('*.json') ))

        for p in itertools.product(['mort_icu', 'los_3'], ['final'], ['l2', 'bce'],
                            [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10]):
            df = dpsgd_df[(dpsgd_df.target == p[0]) & (dpsgd_df.loss == p[2]) 
                            & (dpsgd_df.target_epsilon == p[3])]
            best = df.nlargest(1, 'auc') 
            clean = best.drop(labels = ['accuracy', 'aps', 'auc', 'dist_to_[0,1]', 'f1', 'fpr',
                    'loss', 'model_id', 'prec', 'seed', 'target', 'thresh', 'tpr',
                    'used_gpu', 'noise_multiplier', 'prv_epsilon', 'target_epsilon',
                    'max_grad_norm', 'total_steps', 'eps_error'], axis = 1).to_dict('records')[0]
            hyperparameter_set[p] = (final_n, clean)
            hyperparameter_set[(p[0], 'benchmarks', p[2], p[3])] = (1, hyperparameter_set[p][1])
    
    h_list = []
    for epsilon in epsilons: 
        eps_error = .01 if epsilon >= 1 else max(.0001, epsilon/100) 
        fixed = { "target": target,
                "target_epsilon": epsilon,
                "eps_error": eps_error,
                "delta": 1e-5,
                "loss": loss,
                "use_full_train": use_full_train,
                "used_gpu" : use_gpu
            }
        h_list += _get_h_from_dict((target, h_pass, loss, epsilon), hyperparameter_set, fixed)
    return h_list


def get_h_LR_expm(target, h_pass, epsilons, run, hyperparameter_set, use_full_train, 
                  use_gpu, data_norm, folder=LR_EXPM_RESULTS_FOLDER, final_n = final_n): 
    """called in the run_expm.py and the timing_benchmark_nonprivate.py scripts. 
    This returns the list of hyperparmeters for the kernel function
    """
    if h_pass in [ "final", "benchmarks"]:
        refined_path = Path(folder, f'refined{run}')
        assert refined_path.exists(), f"{refined_path} does not exist!\n h_pass = 'final' and run = {run} requires {refined_path.as_posix()} to exist. (this is created when running run_nonprivate.py with h_pass = 'refined'.)" 
        expm_df = pd.DataFrame.from_records( map( unjsonify,refined_path.rglob('*.json') ))

        for p in itertools.product(['mort_icu', 'los_3'], ['final'], epsilons):
            df = expm_df[(expm_df.target == p[0]) & (expm_df.epsilon == p[2])]
            best = df.nlargest(1, 'auc_ave') 
            clean = best.drop(labels = ['auc_ave', 'model_id', 'seed', 'target', 'epsilon', 'aucs',
                                        'use_full_train'], axis = 1).to_dict('records')[0]
            hyperparameter_set[p] = (final_n, clean)
            hyperparameter_set[(p[0], 'benchmarks', p[2])] = (1, hyperparameter_set[p][1])
    
    fixed = {'target': target,
            's' : 1, # sensitivity 
            "used_gpu" : use_gpu,
            'use_full_train': use_full_train,
            'data_normalization': data_norm}
    h_list = []
    for epsilon in epsilons:
        h_list += [h|{"epsilon": epsilon} for h in _get_h_from_dict((target, h_pass, epsilon), hyperparameter_set, fixed) ]
    return h_list


###### 
## NONPRIVATE HYPERPARAMETERS
#####
non_priv_hyperparameter_set = {
## a key should be of the form:
# (target, h_pass, loss, use_bn)
# target: 'mort_icu' or 'los_3'
# h_pass: any string
# loss: 'l2' or 'bce'
# use_bn: 'bn' or 'nobn'
## for checking this parameters generally do well between 75-85 auc
('mort_icu', 'debug', 'l2') : (1, { 'learning_rate': 0.001,
                                    'lam' : .1,
                                    'momentum': .5,
                                    'num_epochs': 3,
                                    'patience': 1000,
                                    'batch_size': 4500,
                                    'seed': 4771}), 

('mort_icu', 'debugging', 'l2') : (1, { 'learning_rate': 0.003563,
                                    'lam' : .00072,
                                    'momentum': .7,
                                    'num_epochs': 700,
                                    'patience': 200,
                                    'batch_size': 2500,
                                    'seed': 4752}), 


('mort_icu', 'first', 'l2') : (30, DictDist({
                                    'learning_rate': Choice(arange(1e-6, .01, 1e-6)),
#                                    'learning_rate': Choice(arange(1e-7, 1e-5, 1e-7)),
                                    'lam' : Choice(arange(1e-5, .01, 1e-5)),
                                    'momentum': Choice(arange(.1, .9, .1)),
                                    'num_epochs': Choice(arange(50, 750, 50)),
                                    'patience': Choice(arange(100, 1000, 100)),
                                    'batch_size': Choice(arange(500, 7500, 500)),
                                    'seed': Choice(arange(0,10000))})), 


('mort_icu', 'refined', 'l2') : (30, DictDist({'learning_rate': Choice(arange(.004, .006, .0001)),
                                    'lam' : Choice(arange(.0002, .0004, .0001)),
                                    'momentum': Choice(arange(.55, .7, .01)),
                                    'num_epochs': Choice(arange(600, 700, 10)),
                                    'patience': Choice(arange(600, 700, 10)),
                                    'batch_size': Choice(arange(2000, 2500, 50)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'bce') : (30, DictDist({
                                    'patience': Choice(arange(300, 400, 10)),
                                    'batch_size': Choice(arange(2000, 3500, 50)),
                                    'lam' : Choice(arange(.006, .008, .0001)),
                                    'num_epochs': Choice(arange(400, 650, 10)),
                                    'learning_rate': Choice(arange(.003, .004, .0001)),
                                    'momentum': Choice(arange(.4, .55, .01)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'l2') : (30, DictDist({
                                    'patience': Choice(arange(300, 400, 10)),
                                    'batch_size': Choice(arange(2000, 3000, 50)),
                                    'lam' : Choice(arange(.0002, .0004, .0001)),
                                    'num_epochs': Choice(arange(600, 700, 10)),
                                    'learning_rate': Choice(arange(.003, .006, .0001)),
                                    'momentum': Choice(arange(.55, .7, .01)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'bce') : (30, DictDist({
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(1750, 2250, 10)),
                                    'lam' : Choice(arange(.002, .0035, .0001)),
                                    'num_epochs': Choice(arange(500, 600, 10)),
                                    'learning_rate': Choice(arange(.004, .005, .0001)),
                                    'momentum': Choice(arange(.4, .55, .01)),
                                    'seed': Choice(arange(0,10000))})), 

}
## use the same debugging set for all possibilities
for p in itertools.product(['mort_icu', 'los_3'], ['debug'], ['l2', 'bce']):
    non_priv_hyperparameter_set[p] = non_priv_hyperparameter_set[('mort_icu', 'debug', 'l2')]

for p in itertools.product(['mort_icu', 'los_3'], ['first'], ['l2', 'bce']):
    non_priv_hyperparameter_set[p] = non_priv_hyperparameter_set[('mort_icu', 'first', 'l2')]


###### 
## DPSGD HYPERPARAMETERS
#####
dpsgd_hyperparameter_set = {
## a key should be of the form:
# (target, h_pass, loss, use_bn, epsilon)
# target: 'mort_icu' or 'los_3'
# h_pass: any string
# loss: 'l2' or 'bce'
# use_bn: 'bn' or 'nobn'
# epsilon: a number
## for checking this parameters generally do well 
('mort_icu', 'debug', 'l2', .001) : (1, {'learning_rate': 0.004592623182789133,
                                        'num_epochs': 2,
                                        'momentum': .5,
                                        'lam':.001,
                                        'patience': 1000,
                                        'batch_size': 4500,
                                        'seed': 4771}), 
## first search space
('mort_icu', 'first', 'l2', .001) : (30, DictDist({'learning_rate': Choice(arange(.0001, .01, .0001)),
                                    'lam' : Choice(arange(.0001, .01, .0001)),
                                    'momentum': Choice(arange(.1, .9, .05)),
                                    'num_epochs': Choice(arange(50, 750, 50)),
                                    'patience': Choice(arange(100, 1000, 100)),
                                    'batch_size': Choice(arange(1000, 5000, 500)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'l2', .0001) : (30, DictDist({'learning_rate': Choice(arange(.0001, .001, .0001)),
                                    'lam' : Choice(arange(.0005, .0015, .0001)),
                                    'momentum': Choice(arange(.4, .6, .05)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(1000, 2500, 50)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'l2', .01) : (30, DictDist({'learning_rate': Choice(arange(.001, .005, .0001)),
                                    'lam' : Choice(arange(.0005, .0015, .0001)),
                                    'momentum': Choice(arange(.3, .5, .05)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(3000, 4500, 50)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'l2', .1) : (30, DictDist({
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(3000, 4500, 50)),
                                    'lam' : Choice(arange(.003, .006, .0001)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'learning_rate': Choice(arange(.001, .0025, .0001)),
                                    'momentum': Choice(arange(.05, .2, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'l2', 1) : (30, DictDist({
                                    'patience': Choice(arange(200, 300, 10)),
                                    'batch_size': Choice(arange(1000, 2500, 50)),
                                    'lam' : Choice(arange(.003, .006, .0001)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'learning_rate': Choice(arange(.001, .0025, .0001)),
                                    'momentum': Choice(arange(.1, .35, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'l2', 5) : (30, DictDist({
                                    'patience': Choice(arange(800, 900, 10)),
                                    'batch_size': Choice(arange(1000, 2500, 50)),
                                    'lam' : Choice(arange(.003, .006, .0001)),
                                    'num_epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.003, .0045, .0001)),
                                    'momentum': Choice(arange(.1, .35, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'l2', 10) : (30, DictDist({
                                    'patience': Choice(arange(300, 400, 10)),
                                    'batch_size': Choice(arange(1000, 2500, 50)),
                                    'lam' : Choice(arange(.0003, .0015, .0001)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'learning_rate': Choice(arange(.007, .0085, .0001)),
                                    'momentum': Choice(arange(.1, .35, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'bce', .0001) : (30, DictDist({'learning_rate': Choice(arange(.001, .005, .0001)),
                                    'lam' : Choice(arange(.005, .009, .0001)),
                                    'momentum': Choice(arange(.4, .6, .05)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(1000, 2500, 50)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'bce', .01) : (30, DictDist({'learning_rate': Choice(arange(.001, .005, .0001)),
                                    'lam' : Choice(arange(.007, .0085, .0001)),
                                    'momentum': Choice(arange(.1, .3, .05)),
                                    'num_epochs': Choice(arange(50, 200, 10)),
                                    'patience': Choice(arange(600, 700, 10)),
                                    'batch_size': Choice(arange(3000, 4500, 50)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'bce', .1) : (30, DictDist({
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(2000, 3500, 50)),
                                    'lam' : Choice(arange(.001, .004, .0001)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'learning_rate': Choice(arange(.001, .0025, .0001)),
                                    'momentum': Choice(arange(.7, .85, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'bce', 1) : (30, DictDist({
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(1000, 2500, 50)),
                                    'lam' : Choice(arange(.003, .006, .0001)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'learning_rate': Choice(arange(.004, .0065, .0001)),
                                    'momentum': Choice(arange(.35, .5, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'bce', 3.5) : (30, DictDist({
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(2000, 3500, 50)),
                                    'lam' : Choice(arange(.0005, .0015, .0001)),
                                    'num_epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.004, .0065, .0001)),
                                    'momentum': Choice(arange(.5, .65, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'bce', 5) : (30, DictDist({
                                    'patience': Choice(arange(800, 900, 10)),
                                    'batch_size': Choice(arange(1000, 2500, 50)),
                                    'lam' : Choice(arange(.006, .009, .0001)),
                                    'num_epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.0075, .009, .0001)),
                                    'momentum': Choice(arange(.35, .5, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 'bce', 10) : (30, DictDist({
                                    'patience': Choice(arange(300, 400, 10)),
                                    'batch_size': Choice(arange(1000, 2500, 50)),
                                    'lam' : Choice(arange(.004, .0055, .0001)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'learning_rate': Choice(arange(.007, .0085, .0001)),
                                    'momentum': Choice(arange(.65, .8, .05)),
                                    'seed': Choice(arange(0,10000))})), 


('los_3', 'refined', 'l2', .0001) : (30, DictDist({'learning_rate': Choice(arange(.001, .005, .0001)),
                                    'lam' : Choice(arange(.005, .0075, .0001)),
                                    'momentum': Choice(arange(.45, .6, .05)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(2500, 4000, 50)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'l2', .01) : (30, DictDist({'learning_rate': Choice(arange(.001, .005, .0001)),
                                    'lam' : Choice(arange(.0005, .0015, .0001)),
                                    'momentum': Choice(arange(.3, .5, .05)),
                                    'num_epochs': Choice(arange(50, 250, 10)),
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(3000, 4500, 50)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'l2', .1) : (30, DictDist({
                                    'patience': Choice(arange(400, 500, 10)),
                                    'batch_size': Choice(arange(3000, 4500, 50)),
                                    'lam' : Choice(arange(.003, .006, .0001)),
                                    'num_epochs': Choice(arange(200, 400, 10)),
                                    'learning_rate': Choice(arange(.001, .0025, .0001)),
                                    'momentum': Choice(arange(.05, .2, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'l2', 2) : (30, DictDist({
                                    'patience': Choice(arange(200, 300, 10)),
                                    'batch_size': Choice(arange(2000, 4500, 50)),
                                    'lam' : Choice(arange(.005, .009, .0001)),
                                    'num_epochs': Choice(arange(50, 250, 10)),
                                    'learning_rate': Choice(arange(.003, .0045, .0001)),
                                    'momentum': Choice(arange(.5, .65, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'l2', 3.5) : (30, DictDist({
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(1500, 3000, 50)),
                                    'lam' : Choice(arange(.005, .009, .0001)),
                                    'num_epochs': Choice(arange(250, 400, 10)),
                                    'learning_rate': Choice(arange(.002, .0035, .0001)),
                                    'momentum': Choice(arange(.2, .35, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'l2', 7) : (30, DictDist({
                                    'patience': Choice(arange(100, 200, 10)),
                                    'batch_size': Choice(arange(2000, 3500, 50)),
                                    'lam' : Choice(arange(.002, .005, .0001)),
                                    'num_epochs': Choice(arange(100, 250, 10)),
                                    'learning_rate': Choice(arange(.003, .0045, .0001)),
                                    'momentum': Choice(arange(.5, .65, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'bce', .0001) : (30, DictDist({
                                    'patience': Choice(arange(500, 600, 10)),
                                    'batch_size': Choice(arange(3000, 4500, 50)),
                                    'lam' : Choice(arange(.002, .005, .0001)),
                                    'num_epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.007, .0085, .0001)),
                                    'momentum': Choice(arange(.4, .55, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'bce', .5) : (30, DictDist({
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(1500, 3000, 50)),
                                    'lam' : Choice(arange(.003, .0055, .0001)),
                                    'num_epochs': Choice(arange(50, 200, 10)),
                                    'learning_rate': Choice(arange(.003, .0053, .0001)),
                                    'momentum': Choice(arange(.4, .55, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'bce', 1) : (30, DictDist({
                                    'patience': Choice(arange(400, 500, 10)),
                                    'batch_size': Choice(arange(1000, 2500, 50)),
                                    'lam' : Choice(arange(.004, .007, .0001)),
                                    'num_epochs': Choice(arange(50, 200, 10)),
                                    'learning_rate': Choice(arange(.004, .0065, .0001)),
                                    'momentum': Choice(arange(.5, .65, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'bce', 2) : (30, DictDist({
                                    'patience': Choice(arange(700, 800, 10)),
                                    'batch_size': Choice(arange(2500, 4000, 50)),
                                    'lam' : Choice(arange(.003, .006, .0001)),
                                    'num_epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.004, .0065, .0001)),
                                    'momentum': Choice(arange(.3, .45, .05)),
                                    'seed': Choice(arange(0,10000))})), 


('los_3', 'refined', 'bce', 5) : (30, DictDist({
                                    'patience': Choice(arange(100, 200, 10)),
                                    'batch_size': Choice(arange(2500, 4000, 50)),
                                    'lam' : Choice(arange(.004, .008, .0001)),
                                    'num_epochs': Choice(arange(50, 200, 10)),
                                    'learning_rate': Choice(arange(.0055, .007, .0001)),
                                    'momentum': Choice(arange(.45, .6, .05)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 'bce', 10) : (30, DictDist({
                                    'patience': Choice(arange(300, 400, 10)),
                                    'batch_size': Choice(arange(1000, 2500, 50)),
                                    'lam' : Choice(arange(.004, .0065, .0001)),
                                    'num_epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.004, .0065, .0001)),
                                    'momentum': Choice(arange(.65, .8, .05)),
                                    'seed': Choice(arange(0,10000))})), 


}

## use the same debugging set for all possibilities
for p in itertools.product(['mort_icu', 'los_3'], ['debug'], ['l2', 'bce'], 
                           [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10]):
    dpsgd_hyperparameter_set[p] = dpsgd_hyperparameter_set[('mort_icu', 'debug', 'l2', .001)]

for p in itertools.product(['mort_icu', 'los_3'], ['first'], ['l2', 'bce'], 
                           [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10]):
    dpsgd_hyperparameter_set[p] = dpsgd_hyperparameter_set[('mort_icu', 'first', 'l2', .001)]

dpsgd_hyperparameter_set[('mort_icu', 'refined', 'l2', .001)] = dpsgd_hyperparameter_set[('mort_icu', 'refined', 'l2', .0001)]
dpsgd_hyperparameter_set[('mort_icu', 'refined', 'l2', .5)] = dpsgd_hyperparameter_set[('mort_icu', 'refined', 'l2', .1)]
dpsgd_hyperparameter_set[('mort_icu', 'refined', 'l2', 2)] = dpsgd_hyperparameter_set[('mort_icu', 'refined', 'l2', 1)]
dpsgd_hyperparameter_set[('mort_icu', 'refined', 'l2', 3.5)] = dpsgd_hyperparameter_set[('mort_icu', 'refined', 'l2', 1)]
dpsgd_hyperparameter_set[('mort_icu', 'refined', 'l2', 7)] = dpsgd_hyperparameter_set[('mort_icu', 'refined', 'l2', 5)]
dpsgd_hyperparameter_set[('mort_icu', 'refined', 'bce', .001)] = dpsgd_hyperparameter_set[('mort_icu', 'refined', 'bce', .0001)]
dpsgd_hyperparameter_set[('mort_icu', 'refined', 'bce', .5)] = dpsgd_hyperparameter_set[('mort_icu', 'refined', 'bce', .1)]
dpsgd_hyperparameter_set[('mort_icu', 'refined', 'bce', 2)] = dpsgd_hyperparameter_set[('mort_icu', 'refined', 'bce', 1)]
dpsgd_hyperparameter_set[('mort_icu', 'refined', 'bce', 7)] = dpsgd_hyperparameter_set[('mort_icu', 'refined', 'bce', 5)]

dpsgd_hyperparameter_set[('los_3', 'refined', 'l2', .001)] = dpsgd_hyperparameter_set[('los_3', 'refined', 'l2', .0001)]
dpsgd_hyperparameter_set[('los_3', 'refined', 'l2', .5)] = dpsgd_hyperparameter_set[('los_3', 'refined', 'l2', .1)]
dpsgd_hyperparameter_set[('los_3', 'refined', 'l2', 1)] = dpsgd_hyperparameter_set[('los_3', 'refined', 'l2', 2)]
dpsgd_hyperparameter_set[('los_3', 'refined', 'l2', 5)] = dpsgd_hyperparameter_set[('los_3', 'refined', 'l2', 3.5)]
dpsgd_hyperparameter_set[('los_3', 'refined', 'l2', 10)] = dpsgd_hyperparameter_set[('los_3', 'refined', 'l2', 7)]
dpsgd_hyperparameter_set[('los_3', 'refined', 'bce', .001)] = dpsgd_hyperparameter_set[('los_3', 'refined', 'bce', .0001)]
dpsgd_hyperparameter_set[('los_3', 'refined', 'bce', .01)] = dpsgd_hyperparameter_set[('los_3', 'refined', 'bce', .0001)]
dpsgd_hyperparameter_set[('los_3', 'refined', 'bce', .1)] = dpsgd_hyperparameter_set[('los_3', 'refined', 'bce', .0001)]
dpsgd_hyperparameter_set[('los_3', 'refined', 'bce', 3.5)] = dpsgd_hyperparameter_set[('los_3', 'refined', 'bce', 2)]
dpsgd_hyperparameter_set[('los_3', 'refined', 'bce', 7)] = dpsgd_hyperparameter_set[('los_3', 'refined', 'bce', 5)]


###### 
## EXPM HYPERPARAMETERS
#####

expm_hyperparameter_set = {
## a key should be of the form:
# (target, h_pass, epsilon)
# target: 'mort_icu' or 'los_3'
# h_pass: any string
# epsilon: a number
## for checking this parameters generally do well between 75-85 auc
('mort_icu', 'debug', 10) : (3, {'n_flows': 4,
                                    'epochs' : 5, # num of times thru the data 
                                    'learning_rate' : .003, #.0001, # step size
                                    'momentum' : .45, #.5, # how much of previous steps' directions to use
                                    'lam' : .08, # regularization constant
                                    'batch_size': 189,
                                    'm' : 1, # Sylvester flow parameter (A is z_size by m)
                                    'hh' : 1, # number of Householder vectors to use
                                    'sample_std': .01,
                                    'patience': 1000,
                                    'seed': 4771}), 

('mort_icu', 'first', 10) : (30, DictDist({'n_flows': Choice(arange(2, 6)),
                                    'learning_rate': Choice(arange(.0001, .01, .0001)),
                                    'lam' : Choice(arange(.0001, .01, .0001)),
                                    'momentum': Choice(arange(.1, .9, .05)),
                                    'epochs': Choice(arange(20, 150, 10)),
                                    'patience': Choice(arange(100, 5000, 250)),
                                    'm' : Choice(arange(1,5)),
                                    'hh' : Choice(arange(1,5)),
                                    'sample_std': Choice(arange(.0001, .1, .0001)),
                                    'batch_size': Choice(arange(100, 500, 50)),
                                    'seed': Choice(arange(0,10000))})), 


('mort_icu', 'refined', 1e-5) : (30, DictDist({'n_flows': Choice(arange(2, 4)),
                                    'epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.0003, .002, .0001)),
                                    'momentum': Choice(arange(.6, .75, .01)),
                                    'lam' : Choice(arange(.005, .007, .0001)),
                                    'batch_size': Choice(arange(200, 300, 10)),
                                    'm' : Choice(arange(1, 3)),
                                    'hh' : Choice(arange(2, 4)),
                                    'sample_std': Choice(arange(.001, .015, .0001)),
                                    'patience': Choice(arange(3000, 4500, 10)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 1e-3) : (30, DictDist({'n_flows': Choice(arange(4, 6)),
                                    'epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.002, .0035, .0001)),
                                    'momentum': Choice(arange(.3, .45, .01)),
                                    'lam' : Choice(arange(.0075, .009, .0001)),
                                    'batch_size': Choice(arange(200, 300, 10)),
                                    'm' : Choice(arange(1, 3)),
                                    'hh' : Choice(arange(1, 3)),
                                    'sample_std': Choice(arange(.001, .005, .0001)),
                                    'patience': Choice(arange(1000, 2500, 10)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', .1) : (30, DictDist({'n_flows': Choice(arange(4, 6)),
                                    'epochs': Choice(arange(200, 350, 10)),
                                    'learning_rate': Choice(arange(.0035, .005, .0001)),
                                    'momentum': Choice(arange(.35, .5, .01)),
                                    'lam' : Choice(arange(.005, .007, .0001)),
                                    'batch_size': Choice(arange(200, 300, 10)),
                                    'm' : Choice(arange(2, 4)),
                                    'hh' : Choice(arange(2, 4)),
                                    'sample_std': Choice(arange(.01, .03, .0001)),
                                    'patience': Choice(arange(800, 2000, 10)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 1) : (30, DictDist({'n_flows': Choice(arange(2, 4)),
                                    'epochs': Choice(arange(200, 350, 10)),
                                    'learning_rate': Choice(arange(.0007, .002, .0001)),
                                    'momentum': Choice(arange(.55, .75, .01)),
                                    'lam' : Choice(arange(.006, .009, .0001)),
                                    'batch_size': Choice(arange(200, 350, 10)),
                                    'm' : Choice(arange(1, 3)),
                                    'hh' : Choice(arange(3, 4)),
                                    'sample_std': Choice(arange(.001, .005, .0001)),
                                    'patience': Choice(arange(100, 1000, 10)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 3.5) : (30, DictDist({'n_flows': Choice(arange(2, 4)),
                                    'epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.005, .007, .0001)),
                                    'momentum': Choice(arange(.1, .35, .01)),
                                    'lam' : Choice(arange(.003, .006, .0001)),
                                    'batch_size': Choice(arange(200, 350, 10)),
                                    'm' : Choice(arange(2, 4)),
                                    'hh' : Choice(arange(1, 2)),
                                    'sample_std': Choice(arange(.003, .01, .0001)),
                                    'patience': Choice(arange(1700, 2500, 10)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 7) : (30, DictDist({'n_flows': Choice(arange(4, 6)),
                                    'epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.006, .0085, .0001)),
                                    'momentum': Choice(arange(.55, .7, .01)),
                                    'lam' : Choice(arange(.005, .009, .0001)),
                                    'batch_size': Choice(arange(200, 350, 10)),
                                    'm' : Choice(arange(1, 3)),
                                    'hh' : Choice(arange(2, 4)),
                                    'sample_std': Choice(arange(.01, .02, .0001)),
                                    'patience': Choice(arange(2500, 3500, 10)),
                                    'seed': Choice(arange(0,10000))})), 

('mort_icu', 'refined', 10) : (30, DictDist({'n_flows': Choice(arange(4, 6)),
                                    'epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.001, .0025, .0001)),
                                    'momentum': Choice(arange(.55, .7, .01)),
                                    'lam' : Choice(arange(.002, .005, .0001)),
                                    'batch_size': Choice(arange(100, 250, 10)),
                                    'm' : Choice(arange(1, 3)),
                                    'hh' : Choice(arange(1, 3)),
                                    'sample_std': Choice(arange(.004, .008, .0001)),
                                    'patience': Choice(arange(100, 1000, 10)),
                                    'seed': Choice(arange(0,10000))})), 


('los_3', 'refined', 1e-5) : (30, DictDist({'n_flows': Choice(arange(2, 4)),
                                    'epochs': Choice(arange(250, 350, 10)),
                                    'learning_rate': Choice(arange(.0003, .002, .0001)),
                                    'momentum': Choice(arange(.6, .75, .01)),
                                    'lam' : Choice(arange(.005, .007, .0001)),
                                    'batch_size': Choice(arange(200, 350, 10)),
                                    'm' : Choice(arange(1, 3)),
                                    'hh' : Choice(arange(2, 4)),
                                    'sample_std': Choice(arange(.001, .01, .0001)),
                                    'patience': Choice(arange(1, 1000, 10)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 1e-3) : (30, DictDist({'n_flows': Choice(arange(4, 6)),
                                    'epochs': Choice(arange(50, 200, 10)),
                                    'learning_rate': Choice(arange(.002, .0035, .0001)),
                                    'momentum': Choice(arange(.3, .45, .01)),
                                    'lam' : Choice(arange(.0075, .009, .0001)),
                                    'batch_size': Choice(arange(200, 300, 10)),
                                    'm' : Choice(arange(1, 3)),
                                    'hh' : Choice(arange(2, 4)),
                                    'sample_std': Choice(arange(.0008, .005, .0001)),
                                    'patience': Choice(arange(1000, 2500, 10)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', .5) : (30, DictDist({'n_flows': Choice(arange(4, 6)),
                                    'epochs': Choice(arange(150, 300, 10)),
                                    'learning_rate': Choice(arange(.001, .004, .0001)),
                                    'momentum': Choice(arange(.2, .45, .01)),
                                    'lam' : Choice(arange(.001, .004, .0001)),
                                    'batch_size': Choice(arange(300, 400, 10)),
                                    'm' : Choice(arange(2, 4)),
                                    'hh' : Choice(arange(2, 4)),
                                    'sample_std': Choice(arange(.005, .01, .0001)),
                                    'patience': Choice(arange(2500, 4000, 10)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 1) : (30, DictDist({'n_flows': Choice(arange(2, 4)),
                                    'epochs': Choice(arange(200, 350, 10)),
                                    'learning_rate': Choice(arange(.0007, .002, .0001)),
                                    'momentum': Choice(arange(.55, .75, .01)),
                                    'lam' : Choice(arange(.006, .009, .0001)),
                                    'batch_size': Choice(arange(200, 350, 10)),
                                    'm' : Choice(arange(1, 3)),
                                    'hh' : Choice(arange(3, 4)),
                                    'sample_std': Choice(arange(.001, .005, .0001)),
                                    'patience': Choice(arange(100, 1000, 10)),
                                    'seed': Choice(arange(0,10000))})), 

('los_3', 'refined', 5) : (30, DictDist({'n_flows': Choice(arange(2, 4)),
                                    'epochs': Choice(arange(250, 400, 10)),
                                    'learning_rate': Choice(arange(.0005, .001, .0001)),
                                    'momentum': Choice(arange(.5, .7, .01)),
                                    'lam' : Choice(arange(.005, .009, .0001)),
                                    'batch_size': Choice(arange(150, 300, 10)),
                                    'm' : Choice(arange(2, 3)),
                                    'hh' : Choice(arange(1, 4)),
                                    'sample_std': Choice(arange(.003, .01, .0001)),
                                    'patience': Choice(arange(2500, 4000, 10)),
                                    'seed': Choice(arange(0,10000))})), 


('los_3', 'refined', 7) : (30, DictDist({'n_flows': Choice(arange(4, 6)),
                                    'epochs': Choice(arange(50, 200, 10)),
                                    'learning_rate': Choice(arange(.002, .005, .0001)),
                                    'momentum': Choice(arange(.35, .5, .01)),
                                    'lam' : Choice(arange(.005, .009, .0001)),
                                    'batch_size': Choice(arange(200, 350, 10)),
                                    'm' : Choice(arange(1, 3)),
                                    'hh' : Choice(arange(2, 4)),
                                    'sample_std': Choice(arange(.001, .005, .0001)),
                                    'patience': Choice(arange(2000, 3500, 10)),
                                    'seed': Choice(arange(0,10000))})) 
}

## use the same debugging set for all possibilities
for p in itertools.product(['mort_icu', 'los_3'], ['debug'], [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10]):
    expm_hyperparameter_set[p] = expm_hyperparameter_set[('mort_icu', 'debug', 10)]

for p in itertools.product(['mort_icu', 'los_3'], ['first'], 
                           [1e-7, 1e-6, 1e-5, .0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10]):
    expm_hyperparameter_set[p] = expm_hyperparameter_set[('mort_icu', 'first', 10)]

expm_hyperparameter_set[('mort_icu', 'refined', 1e-6)] = expm_hyperparameter_set[('mort_icu', 'refined', 1e-5)]
expm_hyperparameter_set[('mort_icu', 'refined', 1e-4)] = expm_hyperparameter_set[('mort_icu', 'refined', 1e-5)]
expm_hyperparameter_set[('mort_icu', 'refined', 1e-2)] = expm_hyperparameter_set[('mort_icu', 'refined', 1e-3)]
expm_hyperparameter_set[('mort_icu', 'refined', .5)] = expm_hyperparameter_set[('mort_icu', 'refined', .1)]
expm_hyperparameter_set[('mort_icu', 'refined', 2)] = expm_hyperparameter_set[('mort_icu', 'refined', 1)]
expm_hyperparameter_set[('mort_icu', 'refined', 5)] = expm_hyperparameter_set[('mort_icu', 'refined', 3.5)]

expm_hyperparameter_set[('los_3', 'refined', 1e-6)] = expm_hyperparameter_set[('los_3', 'refined', 1e-5)]
expm_hyperparameter_set[('los_3', 'refined', 1e-4)] = expm_hyperparameter_set[('los_3', 'refined', 1e-3)]
expm_hyperparameter_set[('los_3', 'refined', 1e-2)] = expm_hyperparameter_set[('los_3', 'refined', 1e-3)]
expm_hyperparameter_set[('los_3', 'refined', .1)] = expm_hyperparameter_set[('los_3', 'refined', .5)]
expm_hyperparameter_set[('los_3', 'refined', 2)] = expm_hyperparameter_set[('los_3', 'refined', 1)]
expm_hyperparameter_set[('los_3', 'refined', 3.5)] = expm_hyperparameter_set[('los_3', 'refined', 1)]
expm_hyperparameter_set[('los_3', 'refined', 10)] = expm_hyperparameter_set[('los_3', 'refined', 7)]

 



















