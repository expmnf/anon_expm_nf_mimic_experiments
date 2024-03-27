
import sys, itertools
from pathlib import Path
# add our ./code/logistic_regression_experiment/ folder so we can find our config file 
sys.path.append(Path(".", "code", "mimic_experiment").as_posix())
sys.path.append(Path(".", "code").as_posix())

from data_utils import *
from experiment_utils import *
from config import * 

final_n = 10

def get_results_df(RESULTS_FOLDER, h_pass, run, task, verbose = False):
    task_d = {}
    i = 0
    folder = Path(RESULTS_FOLDER, f"{h_pass}{run}")
    for filename in folder.glob('*'):
        if os.path.isdir(filename):
            for subfilename in filename.glob('*'):
                if task in str(filename) and 'json' in str(subfilename) and 'results' in str(subfilename):
                    task_d[i] = unjsonify(subfilename)
                    i += 1
        if task in str(filename) and 'json' in str(filename):
            task_d[i] = unjsonify(filename)
            i += 1
    if verbose: print(f'---Processing {h_pass}{run} run for {task} ICU hyperparameter results -----')
    task_df = pd.concat([pd.json_normalize(task_d[j]) for j in range(0,i)])
    return task_df

def _get_h_from_dict(h_key, hyperparameter_set, fixed):
    # get hyperparameter set from dictionary above
    # TODO make this work for hkey
    N, GRU_D_dist = hyperparameter_set[h_key]
    if type(GRU_D_dist) == DictDist:
        GRU_D_hyperparams_list = GRU_D_dist.rvs(N)
    else:
        # get seeds first
        GRU_D_hyperparams_list = DictDist({'seed': ss.randint(1, 10000),}).rvs(N)
        for h in GRU_D_hyperparams_list:
            h.update(GRU_D_dist)
            
    model_id = 0
    for h in GRU_D_hyperparams_list:
        h.update(fixed)
        model_id += 1
        h.update({'model_id': model_id})

    return GRU_D_hyperparams_list

def get_h_grud_nonpriv(target, h_pass, loss, use_bn, run, hyperparameter_set, use_full_train,
                       use_gpu, run_folder, folder = GRUD_BASELINE_RESULTS_FOLDER, final_n = final_n):
    ## AUTOMATICALLY get best hyperparameters for final run
    if h_pass in ["final", "benchmarks"]:
        refined_path = Path(folder, f'refined{run}')
        assert refined_path.exists(), f"{refined_path} does not exist! h_pass = 'final' and run = {run} requires {refined_path.as_posix()} to exist (this is created with h_pass = 'refined')"
        nonpriv_refined_df = get_results_df(folder, 'refined', 0, target)
        p = (target, 'final', loss, use_bn)
        task_df = nonpriv_refined_df[(nonpriv_refined_df.use_bn == p[3]) & (nonpriv_refined_df.loss == p[2])]
        best = task_df.nlargest(1, 'auc') 
        clean = best.drop(labels = ['accuracy', 'aps', 'auc', 'dist_to_[0,1]', 'f1', 'fpr',
               'loss', 'model_id', 'prec', 'seed', 'target', 'thresh', 'tpr',
               'use_full_train', 'used_gpu'], axis = 1).to_dict('records')[0]        
        print(clean)
        non_priv_hyperparameter_set[p] = (final_n, clean)
        non_priv_hyperparameter_set[(p[0], 'benchmarks', p[2], p[3])] = (1, non_priv_hyperparameter_set[p][1])

    # add fixed hyperparameters
    fixed = {"target": target,
             "use_full_train": use_full_train,
             "loss": loss,
             "use_bn" : use_bn,
             "use_full_train": use_full_train,
             "used_gpu" : use_gpu}

    results_folder = Path(run_folder, f"{target}_loss_{loss}_{use_bn}_results")
    results_folder.mkdir(exist_ok= True)  
    return _get_h_from_dict((target, h_pass, loss, use_bn), hyperparameter_set, fixed)

# get the hyperparameters corresponding to this target and pass (defined above)        
def get_h_grud_dpsgd(target, h_pass, loss, use_bn, epsilon, delta,
                     run, hyperparameter_set, use_full_train,
                     use_gpu, run_folder,
                     folder = GRUD_DPSGD_RESULTS_FOLDER, final_n = final_n):

    ## AUTOMATICALLY get best hyperparameters for final run
    if h_pass in ['final', 'benchmarks']:
        refined_path = Path(folder, f'refined{run}')
        assert refined_path.exists(), f"{refined_path} does not exist! h_pass = 'final' and run = {run} requires {refined_path.as_posix()} to exist (this is created with h_pass = 'refined')"
        dpsgd_refined_df = get_results_df(folder, 'refined', 0, target)
        p = (target, 'final', loss, use_bn, epsilon)
        task_df = dpsgd_refined_df[(dpsgd_refined_df.use_bn == use_bn) & (dpsgd_refined_df.loss == loss) & (dpsgd_refined_df.target_epsilon == epsilon)]
        best = task_df.nlargest(1, 'auc') 
        clean = best.drop(labels = ['accuracy', 'aps', 'auc', 'dist_to_[0,1]', 'f1', 'fpr',
               'loss', 'model_id', 'prec', 'seed', 'target', 'thresh', 'tpr',
               'use_full_train', 'used_gpu'], axis = 1).to_dict('records')[0]        
        dpsgd_hyperparameter_set[p] = (final_n, clean)
        dpsgd_hyperparameter_set[(p[0], 'benchmarks', p[2], p[3], p[4])] = (1, dpsgd_hyperparameter_set[p][1])

    eps_error = .01 if epsilon >= 1 else max(.0001, epsilon/100) 
    # add fixed hyperparameters
    fixed = {"target": target,
             "target_epsilon": epsilon,
             "eps_error": eps_error,
             "delta": delta,
             "loss": loss,
             "use_full_train": use_full_train,
             "use_bn" : use_bn,
             "used_gpu" : use_gpu}

    results_folder = Path(run_folder, f"{target}_epsilon_{epsilon}_loss_{loss}_{use_bn}_results")
    results_folder.mkdir(exist_ok= True)  
    return _get_h_from_dict((target, h_pass, loss, use_bn, epsilon), hyperparameter_set, fixed)

# get the hyperparameters corresponding to this target and pass (defined above)        
def get_h_grud_expm(target, h_pass, epsilon, use_gpu, run, hyperparameter_set, use_full_train,
                    run_folder, folder = GRUD_EXPM_RESULTS_FOLDER, final_n = final_n):
      
    ## AUTOMATICALLY get the best hyperparameter results for final run
    if h_pass in ['final', 'benchmarks']:
        refined_path = Path(folder, f'refined{run}')
        assert refined_path.exists(), f"{refined_path} does not exist! h_pass = 'final' and run = {run} requires {refined_path.as_posix()} to exist (this is created with h_pass = 'refined')"
        expm_refined_df = get_results_df(folder, 'refined', run, 'mort_icu')
        #for p in itertools.product([target], ['final'], [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10]):
        p = (target, 'final', epsilon)
        task_df = expm_refined_df[expm_refined_df.epsilon == epsilon]
        best = task_df.nlargest(1, 'auc_ave') 
        clean = best.drop(labels = ['aucs', 
                    #'dist_to_[0,1]', 'f1', 'fpr', 'thresh', 'tpr', 'aps', 'prec', 'accuracy',
                   'model_id', 'seed', 'target', 'auc_ave',
                   'use_full_train', 'used_gpu'], axis = 1).to_dict('records')[0]        

        expm_hyperparameter_set[p] = (final_n, clean)
        expm_hyperparameter_set[(p[0], 'benchmarks', p[2])] = (1, expm_hyperparameter_set[p][1])

    fixed = {'target': target,
          'epsilon' : epsilon, # privacy
          's' : 1, # sensitivity 
          'use_full_train': use_full_train,
          #'skip': 5, # how long to wait before recording loss during training 
          'used_gpu' : use_gpu}

    ## make subfolder for results: 
    results_folder = Path(run_folder, f'{target}_epsilon_{epsilon}_results')
    results_folder.mkdir(exist_ok= True)  
    return _get_h_from_dict((target, h_pass, epsilon), hyperparameter_set, fixed)

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
('mort_icu', 'debug', 'l2', 'bn') : (1, {'hidden_size': 80,
                                        'learning_rate': 0.004592623182789133,
                                        'num_epochs': 10,
                                        'patience': 1000,
                                        'batch_size': 53,
                                        'dropout_p' : .5,
                                        'seed': 4771}), 

## first search space
#('mort_icu', 'first', 'l2', 'bn') : (30, DictDist({'hidden_size': ss.randint(10, 100), 
('mort_icu', 'first', 'l2', 'bn') : (30, DictDist({'hidden_size': ss.randint(10, 100), 
                                                'learning_rate': ss.uniform(.0001, .1),
                                                'num_epochs': ss.randint(2, 50),
                                                'patience': ss.randint(1000, 5000),
                                                'batch_size': ss.randint(25, 100),
                                                'dropout_p' : ss.uniform(0, .5),
                                                'seed': ss.randint(1, 10000),})), 


('mort_icu', 'refined', 'l2', 'bn') : (15, DictDist({'hidden_size': ss.randint(30, 40), 
                                                'learning_rate': ss.uniform(.0005, .002),
                                                'num_epochs': ss.randint(3, 15),
                                                'patience': ss.randint(4000, 4200),
                                                'batch_size': ss.randint(80, 90),
                                                'dropout_p' : ss.uniform(.05, .1),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'bce', 'bn') : (15, DictDist({'hidden_size': ss.randint(25, 35), 
                                                'learning_rate': ss.uniform(.001, .002),
                                                'num_epochs': ss.randint(3, 10),
                                                'patience': ss.randint(900, 1100),
                                                'batch_size': ss.randint(70, 80),
                                                'dropout_p' : ss.uniform(.5, .1),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'l2', 'nobn') : (15, DictDist({'hidden_size': ss.randint(30, 40), 
                                                'learning_rate': ss.uniform(.0005, .002),
                                                'num_epochs': ss.randint(3, 15),
                                                'patience': ss.randint(4000, 4200),
                                                'batch_size': ss.randint(80, 90),
                                                'dropout_p' : ss.uniform(.05, .1),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'bce', 'nobn') : (15, DictDist({'hidden_size': ss.randint(30, 40), 
                                                'learning_rate': ss.uniform(.0005, .002),
                                                'num_epochs': ss.randint(3, 15),
                                                'patience': ss.randint(4000, 4200),
                                                'batch_size': ss.randint(80, 90),
                                                'dropout_p' : ss.uniform(.05, .1),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'l2', 'bn') : (15, DictDist({'hidden_size': ss.randint(25, 35), 
                                                'learning_rate': ss.uniform(.001, .002),
                                                'num_epochs': ss.randint(3, 10),
                                                'patience': ss.randint(900, 1100),
                                                'batch_size': ss.randint(70, 80),
                                                'dropout_p' : ss.uniform(.5, .1),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'bce', 'bn') : (15, DictDist({'hidden_size': ss.randint(25, 35), 
                                                'learning_rate': ss.uniform(.001, .002),
                                                'num_epochs': ss.randint(3, 10),
                                                'patience': ss.randint(900, 1100),
                                                'batch_size': ss.randint(70, 80),
                                                'dropout_p' : ss.uniform(.5, .1),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'l2', 'nobn') : (15, DictDist({'hidden_size': ss.randint(30, 40), 
                                                'learning_rate': ss.uniform(.0005, .002),
                                                'num_epochs': ss.randint(3, 15),
                                                'patience': ss.randint(4000, 4200),
                                                'batch_size': ss.randint(80, 90),
                                                'dropout_p' : ss.uniform(.05, .1),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'bce', 'nobn') : (15, DictDist({'hidden_size': ss.randint(30, 40), 
                                                'learning_rate': ss.uniform(.0005, .002),
                                                'num_epochs': ss.randint(3, 15),
                                                'patience': ss.randint(4000, 4200),
                                                'batch_size': ss.randint(80, 90),
                                                'dropout_p' : ss.uniform(.05, .1),
                                                'seed': ss.randint(1, 10000),})), 



}
## use the same debugging set for all possibilities
for p in itertools.product(['mort_icu', 'los_3'], ['debug'], ['l2', 'bce'], ['bn', 'nobn']):
    non_priv_hyperparameter_set[p] = non_priv_hyperparameter_set[('mort_icu', 'debug', 'l2', 'bn')]

## use the same first pass search
for p in itertools.product(['mort_icu', 'los_3'], ['first'], ['l2', 'bce'], ['bn', 'nobn']):
    non_priv_hyperparameter_set[p] = non_priv_hyperparameter_set[('mort_icu', 'first', 'l2', 'bn')]




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
('mort_icu', 'debug', 'l2', 'nobn', .001) : (3, {'hidden_size': 10,
                                        'learning_rate': 0.004592623182789133,
                                        'num_epochs': 2,
                                        'patience': 1000,
                                        'batch_size': 65,
                                        'dropout_p' : .5,
                                        'seed': 4771}), 
## first search space
('mort_icu', 'first', 'l2', 'nobn', 10) : (30, DictDist({'hidden_size': ss.randint(10, 100), 
                                                'learning_rate': ss.uniform(.002, .1),
                                                'num_epochs': ss.randint(2, 25),
                                                'patience': ss.randint(600, 4000),
                                                'batch_size': ss.randint(35, 65),
                                                'dropout_p' : ss.uniform(0, .5),
                                                'seed': ss.randint(1, 10000),})),

## refined search space
('mort_icu', 'refined', 'l2', 'nobn', 7) : (15, DictDist({'hidden_size': ss.randint(60, 65), 
                                                'learning_rate': ss.uniform(.002, .005),
                                                'num_epochs': ss.randint(7, 11),
                                                'patience': ss.randint(3500, 3750),
                                                'batch_size': ss.randint(55, 60),
                                                'dropout_p' : ss.uniform(.001, .01),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'bce', 'nobn', 7) : (15, DictDist({'hidden_size': ss.randint(52, 57), 
                                                'learning_rate': ss.uniform(.001, .0025),
                                                'num_epochs': ss.randint(5, 11),
                                                'patience': ss.randint(800, 900),
                                                'batch_size': ss.randint(35, 45),
                                                'dropout_p' : ss.uniform(.15, .1),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'l2', 'nobn', 2) : (15, DictDist({'hidden_size': ss.randint(60, 65), 
                                                'learning_rate': ss.uniform(.001, .005),
                                                'num_epochs': ss.randint(5, 11),
                                                'patience': ss.randint(900, 1000),
                                                'batch_size': ss.randint(35, 45),
                                                'dropout_p' : ss.uniform(.1, .2),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'bce', 'nobn', 2) : (15, DictDist({'hidden_size': ss.randint(7, 20), 
                                                'learning_rate': ss.uniform(.001, .0025),
                                                'num_epochs': ss.randint(20, 30),
                                                'patience': ss.randint(3400, 3600),
                                                'batch_size': ss.randint(54, 58),
                                                'dropout_p' : ss.uniform(.3, .1),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'l2', 'nobn', .5) : (15, DictDist({'hidden_size': ss.randint(75, 85), 
                                                'learning_rate': ss.uniform(.002, .002),
                                                'num_epochs': ss.randint(5, 11),
                                                'patience': ss.randint(1500, 1700),
                                                'batch_size': ss.randint(35, 45),
                                                'dropout_p' : ss.uniform(.3, .1),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'bce', 'nobn', .5) : (15, DictDist({'hidden_size': ss.randint(55, 65), 
                                                'learning_rate': ss.uniform(.0035, .002),
                                                'num_epochs': ss.randint(7, 11),
                                                'patience': ss.randint(3500, 3700),
                                                'batch_size': ss.randint(55, 65),
                                                'dropout_p' : ss.uniform(.002, .01),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'l2', 'nobn', .1) : (15, DictDist({'hidden_size': ss.randint(65, 75), 
                                                'learning_rate': ss.uniform(.0055, .002),
                                                'num_epochs': ss.randint(11, 15),
                                                'patience': ss.randint(2700, 2900),
                                                'batch_size': ss.randint(45, 55),
                                                'dropout_p' : ss.uniform(.3, .1),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'bce', 'nobn', .1) : (15, DictDist({'hidden_size': ss.randint(35, 45), 
                                                'learning_rate': ss.uniform(.005, .002),
                                                'num_epochs': ss.randint(19, 23),
                                                'patience': ss.randint(3000, 3200),
                                                'batch_size': ss.randint(40, 50),
                                                'dropout_p' : ss.uniform(.45, .1),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'l2', 'nobn', .01) : (15, DictDist({'hidden_size': ss.randint(45, 55), 
                                                'learning_rate': ss.uniform(.0075, .002),
                                                'num_epochs': ss.randint(3, 9),
                                                'patience': ss.randint(1000, 1200),
                                                'batch_size': ss.randint(60, 70),
                                                'dropout_p' : ss.uniform(.002, .01),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'bce', 'nobn', .01) : (15, DictDist({'hidden_size': ss.randint(25, 35), 
                                                'learning_rate': ss.uniform(.005, .015),
                                                'num_epochs': ss.randint(9, 13),
                                                'patience': ss.randint(3400, 3600),
                                                'batch_size': ss.randint(55, 65),
                                                'dropout_p' : ss.uniform(.3, .1),
                                                'seed': ss.randint(1, 10000),})),

('mort_icu', 'refined', 'l2', 'nobn', .001) : (15, DictDist({'hidden_size': ss.randint(15, 25), 
                                                'learning_rate': ss.uniform(.065, .02),
                                                'num_epochs': ss.randint(2, 15),
                                                'patience': ss.randint(1000, 1200),
                                                'batch_size': ss.randint(30, 40),
                                                'dropout_p' : ss.uniform(.2, .1),
                                                'seed': ss.randint(1, 10000),})), 

('mort_icu', 'refined', 'bce', 'nobn', .001) : (15, DictDist({'hidden_size': ss.randint(10, 20), 
                                                'learning_rate': ss.uniform(.04, .025),
                                                'num_epochs': ss.randint(12, 18),
                                                'patience': ss.randint(2800, 3000),
                                                'batch_size': ss.randint(35, 45),
                                                'dropout_p' : ss.uniform(.15, .1),
                                                'seed': ss.randint(1, 10000),})),

('mort_icu', 'refined', 'l2', 'nobn', .0001) : (15, DictDist({'hidden_size': ss.randint(20, 30), 
                                                'learning_rate': ss.uniform(.005, .025),
                                                'num_epochs': ss.randint(10, 14),
                                                'patience': ss.randint(500, 700),
                                                'batch_size': ss.randint(35, 45),
                                                'dropout_p' : ss.uniform(.2, .1),
                                                'seed': ss.randint(1, 10000),})),

('mort_icu', 'refined', 'bce', 'nobn', .0001) : (15, DictDist({'hidden_size': ss.randint(80, 90), 
                                                'learning_rate': ss.uniform(.065, .02),
                                                'num_epochs': ss.randint(11, 15),
                                                'patience': ss.randint(3400, 3600),
                                                'batch_size': ss.randint(30, 40),
                                                'dropout_p' : ss.uniform(.1, .1),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'l2', 'nobn', 2) : (15, DictDist({'hidden_size': ss.randint(50, 60), 
                                                'learning_rate': ss.uniform(.002, .002),
                                                'num_epochs': ss.randint(10, 15),
                                                'patience': ss.randint(1300, 1500),
                                                'batch_size': ss.randint(45, 55),
                                                'dropout_p' : ss.uniform(.28, .1),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'bce', 'nobn', 2) : (15, DictDist({'hidden_size': ss.randint(7, 20), 
                                                'learning_rate': ss.uniform(.001, .0025),
                                                'num_epochs': ss.randint(20, 30),
                                                'patience': ss.randint(3400, 3600),
                                                'batch_size': ss.randint(54, 58),
                                                'dropout_p' : ss.uniform(.3, .1),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'l2', 'nobn', .5) : (15, DictDist({'hidden_size': ss.randint(25, 30), 
                                                'learning_rate': ss.uniform(.003, .0015),
                                                'num_epochs': ss.randint(12, 17),
                                                'patience': ss.randint(2200, 2400),
                                                'batch_size': ss.randint(50, 60),
                                                'dropout_p' : ss.uniform(.01, .2),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'bce', 'nobn', .5) : (15, DictDist({'hidden_size': ss.randint(50, 60), 
                                                'learning_rate': ss.uniform(.008, .002),
                                                'num_epochs': ss.randint(3, 11),
                                                'patience': ss.randint(1100, 1300),
                                                'batch_size': ss.randint(60, 70),
                                                'dropout_p' : ss.uniform(.005, .01),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'l2', 'nobn', .1) : (15, DictDist({'hidden_size': ss.randint(15, 25), 
                                                'learning_rate': ss.uniform(.003, .002),
                                                'num_epochs': ss.randint(11, 15),
                                                'patience': ss.randint(2300, 2500),
                                                'batch_size': ss.randint(50, 60),
                                                'dropout_p' : ss.uniform(.2, .2),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'bce', 'nobn', .1) : (15, DictDist({'hidden_size': ss.randint(20, 30), 
                                                'learning_rate': ss.uniform(.002, .002),
                                                'num_epochs': ss.randint(11, 17),
                                                'patience': ss.randint(2900, 3100),
                                                'batch_size': ss.randint(55, 65),
                                                'dropout_p' : ss.uniform(.2, .2),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'l2', 'nobn', .01) : (15, DictDist({'hidden_size': ss.randint(55, 65), 
                                                'learning_rate': ss.uniform(.007, .02),
                                                'num_epochs': ss.randint(12, 16),
                                                'patience': ss.randint(700, 900),
                                                'batch_size': ss.randint(50, 60),
                                                'dropout_p' : ss.uniform(.02, .04),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'bce', 'nobn', .01) : (15, DictDist({'hidden_size': ss.randint(45, 55), 
                                                'learning_rate': ss.uniform(.007, .005),
                                                'num_epochs': ss.randint(3, 11),
                                                'patience': ss.randint(1000, 1200),
                                                'batch_size': ss.randint(60, 70),
                                                'dropout_p' : ss.uniform(.002, .01),
                                                'seed': ss.randint(1, 10000),})),

('los_3', 'refined', 'l2', 'nobn', .001) : (15, DictDist({'hidden_size': ss.randint(15, 25), 
                                                'learning_rate': ss.uniform(.06, .2),
                                                'num_epochs': ss.randint(12, 16),
                                                'patience': ss.randint(700, 900),
                                                'batch_size': ss.randint(55, 65),
                                                'dropout_p' : ss.uniform(.3, .1),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'bce', 'nobn', .001) : (15, DictDist({'hidden_size': ss.randint(25, 35), 
                                                'learning_rate': ss.uniform(.04, .02),
                                                'num_epochs': ss.randint(7, 11),
                                                'patience': ss.randint(800, 1000),
                                                'batch_size': ss.randint(30, 40),
                                                'dropout_p' : ss.uniform(.07, .05),
                                                'seed': ss.randint(1, 10000),})),

('los_3', 'refined', 'l2', 'nobn', .0001) : (15, DictDist({'hidden_size': ss.randint(15, 25), 
                                                'learning_rate': ss.uniform(.06, .2),
                                                'num_epochs': ss.randint(12, 16),
                                                'patience': ss.randint(700, 900),
                                                'batch_size': ss.randint(60, 70),
                                                'dropout_p' : ss.uniform(.25, .1),
                                                'seed': ss.randint(1, 10000),})), 

('los_3', 'refined', 'bce', 'nobn', .0001) : (15, DictDist({'hidden_size': ss.randint(35, 45), 
                                                'learning_rate': ss.uniform(.08, .02),
                                                'num_epochs': ss.randint(20, 24),
                                                'patience': ss.randint(1300, 1500),
                                                'batch_size': ss.randint(40, 50),
                                                'dropout_p' : ss.uniform(.35, .1),
                                                'seed': ss.randint(1, 10000),})),


}

## use the same debugging set for all possibilities
for p in itertools.product(['mort_icu', 'los_3'], ['debug'], ['l2', 'bce'], ['nobn'], [.0001, .001, .01, .1, 1, 10]):
    dpsgd_hyperparameter_set[p] = dpsgd_hyperparameter_set[('mort_icu', 'debug', 'l2', 'nobn', .001)]

## use the same first set for all possibilities
for p in itertools.product(['mort_icu', 'los_3'], ['first'], ['l2', 'bce'], ['nobn'], 
                           [.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 3.5, 5, 7, 10]):
    dpsgd_hyperparameter_set[p] = dpsgd_hyperparameter_set[('mort_icu', 'first', 'l2', 'nobn', 10)]

# epsilon = 7 did really well, so we'll try near those for epsilons close to 7
# and the best ones were the same for both tasks
for p in itertools.product(['mort_icu', 'los_3'], ['refined'], ['l2', 'bce'], ['nobn'], [5, 7, 10]):
    dpsgd_hyperparameter_set[p] = dpsgd_hyperparameter_set[('mort_icu', 'refined', p[2], 'nobn', 7)]

# epsilon = 2 seemed to find something good so we'll try those for epsilon near 2
for p in itertools.product(['mort_icu', 'los_3'], ['refined'], ['l2', 'bce'], ['nobn'], [1, 3.5]):
    dpsgd_hyperparameter_set[p] = dpsgd_hyperparameter_set[(p[0], 'refined', p[2], 'nobn', 2)]




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
('mort_icu', 'debug', 1) : (2, {'n_flows': 4,
                                    'epochs' : 5, # num of times thru the data 
                                    'learning_rate' : .003, #.0001, # step size
                                    'momentum' : .45, #.5, # how much of previous steps' directions to use
                                    'lam' : .08, # regularization constant
                                    'batch_size': 189,
                                    'm' : 1, # Sylvester flow parameter (A is z_size by m)
                                    'hh' : 1, # number of Householder vectors to use
                                    'sample_std': .01,
                                    'patience': 1000,
                                    # grud parameters
                                    'data_batch_size': 1373,
                                    'hidden_size': 9,
                                    'dropout_p' : .25,
                                    'seed': 4771}), 

## first search space
('mort_icu', 'first', 10) : (30, DictDist({'n_flows': Choice([3,4,5]),
                                    'epochs': ss.randint(15, 50),
                                    'learning_rate': ss.uniform(.0001, .005),
                                    'momentum' : ss.uniform(.1, .7),
                                    'lam' : ss.uniform(0, .3), # regularization constant
                                    'batch_size': ss.randint(100, 200),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.01, .49),
                                    'patience': ss.randint(800, 3000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(750, 1500),
                                    'hidden_size': ss.randint(5, 10),
                                    'dropout_p' : ss.uniform(0, .5),
                                    'seed': ss.randint(0, 10000)})),

## refined search space
('mort_icu', 'refined', 10) : (2, DictDist({'n_flows': Choice([3,5]),
                                    'epochs': ss.randint(40, 50),
                                    'learning_rate': ss.uniform(.001, .002),
                                    'momentum' : ss.uniform(.25, .1),
                                    'lam' : ss.uniform(.005, .01), # regularization constant
                                    'batch_size': ss.randint(100, 120),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.005, .01),
                                    'patience': ss.randint(2500, 3000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1000, 1150),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.2, .1),
                                    'seed': ss.randint(0, 10000)})),

('mort_icu', 'refined', 7) : (15, DictDist({'n_flows': Choice([5]),
                                    'epochs': ss.randint(40, 50),
                                    'learning_rate': ss.uniform(.001, .002),
                                    'momentum' : ss.uniform(.6, .15),
                                    'lam' : ss.uniform(.1, .1), # regularization constant
                                    'batch_size': ss.randint(160, 180),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.01, .05),
                                    'patience': ss.randint(1100, 1300),
                                    # grud parameters
                                    'data_batch_size': ss.randint(800, 1000),
                                    'hidden_size': ss.randint(5, 7),
                                    'dropout_p' : ss.uniform(.02, .1),
                                    'seed': ss.randint(0, 10000)})),


('mort_icu', 'refined', 5) : (15, DictDist({'n_flows': Choice([4,5]),
                                    'epochs': ss.randint(45, 55),
                                    'learning_rate': ss.uniform(.004, .002),
                                    'momentum' : ss.uniform(.15, .1),
                                    'lam' : ss.uniform(.2, .1), # regularization constant
                                    'batch_size': ss.randint(150, 175),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.04, .1),
                                    'patience': ss.randint(2500, 3000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(800, 950),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.4, .1),
                                    'seed': ss.randint(0, 10000)})),

('mort_icu', 'refined', 3.5) : (15, DictDist({'n_flows': Choice([5]),
                                    'epochs': ss.randint(40, 50),
                                    'learning_rate': ss.uniform(.001, .005),
                                    'momentum' : ss.uniform(.55, .1),
                                    'lam' : ss.uniform(.05, .1), # regularization constant
                                    'batch_size': ss.randint(120, 140),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.05, .1),
                                    'patience': ss.randint(1500, 2000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(800, 1000),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.2, .2),
                                    'seed': ss.randint(0, 10000)})),

('mort_icu', 'refined', 2) : (15, DictDist({'n_flows': Choice([3,4,5]),
                                    'epochs': ss.randint(45, 55),
                                    'learning_rate': ss.uniform(.002, .002),
                                    'momentum' : ss.uniform(.3, .2),
                                    'lam' : ss.uniform(.05, .1), # regularization constant
                                    'batch_size': ss.randint(165, 185),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.05, .1),
                                    'patience': ss.randint(1750, 2250),
                                    # grud parameters
                                    'data_batch_size': ss.randint(800, 950),
                                    'hidden_size': ss.randint(8, 10),
                                    'dropout_p' : ss.uniform(.2, .1),
                                    'seed': ss.randint(0, 10000)})),


('mort_icu', 'refined', 1) : (15, DictDist({'n_flows': Choice([4]),
                                    'epochs': ss.randint(25, 30),
                                    'learning_rate': ss.uniform(.004, .001),
                                    'momentum' : ss.uniform(.25, .1),
                                    'lam' : ss.uniform(.01, .01), # regularization constant
                                    'batch_size': ss.randint(100, 120),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.075, .1),
                                    'patience': ss.randint(1500, 2000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1200, 1300),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.005, .1),
                                    'seed': ss.randint(0, 10000)})),

('mort_icu', 'refined', .5) : (15, DictDist({'n_flows': Choice([5]),
                                    'epochs': ss.randint(40, 50),
                                    'learning_rate': ss.uniform(.002, .003),
                                    'momentum' : ss.uniform(.55, .1),
                                    'lam' : ss.uniform(.05, .1), # regularization constant
                                    'batch_size': ss.randint(160, 180),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.05, .1),
                                    'patience': ss.randint(1500, 2000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1000, 1200),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.01, .02),
                                    'seed': ss.randint(0, 10000)})),

('mort_icu', 'refined', .1) : (15, DictDist({'n_flows': Choice([3]),
                                    'epochs': ss.randint(45, 55),
                                    'learning_rate': ss.uniform(.001, .003),
                                    'momentum' : ss.uniform(.3, .15),
                                    'lam' : ss.uniform(.05, .1), # regularization constant
                                    'batch_size': ss.randint(175, 200),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.05, .1),
                                    'patience': ss.randint(750, 1250),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1000, 1200),
                                    'hidden_size': ss.randint(5, 7),
                                    'dropout_p' : ss.uniform(.3, .15),
                                    'seed': ss.randint(0, 10000)})),

('mort_icu', 'refined', .01) : (15, DictDist({'n_flows': Choice([3]),
                                    'epochs': ss.randint(35, 45),
                                    'learning_rate': ss.uniform(.002, .002),
                                    'momentum' : ss.uniform(.6, .15),
                                    'lam' : ss.uniform(.15, .1), # regularization constant
                                    'batch_size': ss.randint(165, 185),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.1, .15),
                                    'patience': ss.randint(2250, 2750),
                                    # grud parameters
                                    'data_batch_size': ss.randint(800, 1000),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.05, .1),
                                    'seed': ss.randint(0, 10000)})),

('mort_icu', 'refined', .001) : (15, DictDist({'n_flows': Choice([4]),
                                    'epochs': ss.randint(35, 45),
                                    'learning_rate': ss.uniform(.003, .002),
                                    'momentum' : ss.uniform(.2, .1),
                                    'lam' : ss.uniform(.05, .1), # regularization constant
                                    'batch_size': ss.randint(130, 150),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.05, .025),
                                    'patience': ss.randint(2500, 3000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1300, 1500),
                                    'hidden_size': ss.randint(6, 8),
                                    'dropout_p' : ss.uniform(.4, .1),
                                    'seed': ss.randint(0, 10000)})),

('mort_icu', 'refined', .0001) : (15, DictDist({'n_flows': Choice([3,4]),
                                    'epochs': ss.randint(15, 25),
                                    'learning_rate': ss.uniform(.002, .003),
                                    'momentum' : ss.uniform(.5, .2),
                                    'lam' : ss.uniform(.1, .15), # regularization constant
                                    'batch_size': ss.randint(140, 160),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.01, .02),
                                    'patience': ss.randint(1500, 2000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(900, 1050),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.1, .2),
                                    'seed': ss.randint(0, 10000)})),

('los_3', 'refined', 10) : (15, DictDist({'n_flows': Choice([5]),
                                    'epochs': ss.randint(15, 25),
                                    'learning_rate': ss.uniform(.002, .002),
                                    'momentum' : ss.uniform(.47, .15),
                                    'lam' : ss.uniform(.05, .1), # regularization constant
                                    'batch_size': ss.randint(180, 200),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.005, .02),
                                    'patience': ss.randint(1200, 1400),
                                    # grud parameters
                                    'data_batch_size': ss.randint(800, 1000),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.06, .15),
                                    'seed': ss.randint(0, 10000)})),

('los_3', 'refined', 7) : (15, DictDist({'n_flows': Choice([4]),
                                    'epochs': ss.randint(15, 25),
                                    'learning_rate': ss.uniform(.003, .002),
                                    'momentum' : ss.uniform(.38, .15),
                                    'lam' : ss.uniform(.05, .1), # regularization constant
                                    'batch_size': ss.randint(160, 185),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.07, .02),
                                    'patience': ss.randint(1500, 1700),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1200, 1400),
                                    'hidden_size': ss.randint(5, 7),
                                    'dropout_p' : ss.uniform(.04, .1),
                                    'seed': ss.randint(0, 10000)})),

('los_3', 'refined', 5) : (15, DictDist({'n_flows': Choice([4, 5]),
                                    'epochs': ss.randint(25, 35),
                                    'learning_rate': ss.uniform(.001, .003),
                                    'momentum' : ss.uniform(.6, .15),
                                    'lam' : ss.uniform(.02, .1), # regularization constant
                                    'batch_size': ss.randint(130, 150),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.01, .04),
                                    'patience': ss.randint(1200, 1400),
                                    # grud parameters
                                    'data_batch_size': ss.randint(800, 1000),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.04, .1),
                                    'seed': ss.randint(0, 10000)})),

('los_3', 'refined', 3.5) : (15, DictDist({'n_flows': Choice([5]),
                                    'epochs': ss.randint(15, 25),
                                    'learning_rate': ss.uniform(.001, .003),
                                    'momentum' : ss.uniform(.6, .15),
                                    'lam' : ss.uniform(.2, .1), # regularization constant
                                    'batch_size': ss.randint(120, 140),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.05, .04),
                                    'patience': ss.randint(800, 1000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1200, 1400),
                                    'hidden_size': ss.randint(6, 8),
                                    'dropout_p' : ss.uniform(.2, .1),
                                    'seed': ss.randint(0, 10000)})),

('los_3', 'refined', 2) : (15, DictDist({'n_flows': Choice([4, 5]),
                                    'epochs': ss.randint(45, 55),
                                    'learning_rate': ss.uniform(.002, .003),
                                    'momentum' : ss.uniform(.3, .15),
                                    'lam' : ss.uniform(.1, .1), # regularization constant
                                    'batch_size': ss.randint(100, 120),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.05, .02),
                                    'patience': ss.randint(1000, 1200),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1400, 1500),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.2, .1),
                                    'seed': ss.randint(0, 10000)})),

('los_3', 'refined', 1) : (15, DictDist({'n_flows': Choice([4]),
                                    'epochs': ss.randint(15, 25),
                                    'learning_rate': ss.uniform(.002, .004),
                                    'momentum' : ss.uniform(.6, .15),
                                    'lam' : ss.uniform(.05, .1), # regularization constant
                                    'batch_size': ss.randint(170, 190),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.01, .02),
                                    'patience': ss.randint(1900, 2100),
                                    # grud parameters
                                    'data_batch_size': ss.randint(800, 1000),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.05, .1),
                                    'seed': ss.randint(0, 10000)})),


('los_3', 'refined', .1) : (15, DictDist({'n_flows': Choice([3]),
                                    'epochs': ss.randint(45, 55),
                                    'learning_rate': ss.uniform(.002, .002),
                                    'momentum' : ss.uniform(.3, .15),
                                    'lam' : ss.uniform(.06, .1), # regularization constant
                                    'batch_size': ss.randint(180, 200),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.04, .02),
                                    'patience': ss.randint(800, 1000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1100, 1300),
                                    'hidden_size': ss.randint(5, 7),
                                    'dropout_p' : ss.uniform(.25, .1),
                                    'seed': ss.randint(0, 10000)})),

('los_3', 'refined', .5) : (15, DictDist({'n_flows': Choice([5]),
                                    'epochs': ss.randint(35, 45),
                                    'learning_rate': ss.uniform(.002, .002),
                                    'momentum' : ss.uniform(.6, .15),
                                    'lam' : ss.uniform(.06, .02), # regularization constant
                                    'batch_size': ss.randint(160, 180),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.05, .1),
                                    'patience': ss.randint(1800, 2000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1000, 1200),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.01, .1),
                                    'seed': ss.randint(0, 10000)})),

('los_3', 'refined', .01) : (15, DictDist({'n_flows': Choice([3]),
                                    'epochs': ss.randint(35, 45),
                                    'learning_rate': ss.uniform(.002, .002),
                                    'momentum' : ss.uniform(.65, .15),
                                    'lam' : ss.uniform(.02, .1), # regularization constant
                                    'batch_size': ss.randint(170, 190),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.1, .1),
                                    'patience': ss.randint(2400, 2600),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1000, 1200),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.4, .1),
                                    'seed': ss.randint(0, 10000)})),

('los_3', 'refined', .001) : (15, DictDist({'n_flows': Choice([4]),
                                    'epochs': ss.randint(30, 40),
                                    'learning_rate': ss.uniform(.002, .002),
                                    'momentum' : ss.uniform(.65, .15),
                                    'lam' : ss.uniform(.002, .01), # regularization constant
                                    'batch_size': ss.randint(160, 180),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.04, .05),
                                    'patience': ss.randint(800, 1000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(1400, 1500),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.4, .1),
                                    'seed': ss.randint(0, 10000)})),

('los_3', 'refined', .0001) : (15, DictDist({'n_flows': Choice([5]),
                                    'epochs': ss.randint(45, 55),
                                    'learning_rate': ss.uniform(.002, .004),
                                    'momentum' : ss.uniform(.2, .15),
                                    'lam' : ss.uniform(.1, .1), # regularization constant
                                    'batch_size': ss.randint(120, 140),
                                    'm' : Choice([1]), # sylvester flow parameter (a is z_size by m)
                                    'hh' : Choice([1]), # number of householder vectors to use
                                    'sample_std': ss.uniform(.01, .02),
                                    'patience': ss.randint(2700, 3000),
                                    # grud parameters
                                    'data_batch_size': ss.randint(800, 1000),
                                    'hidden_size': ss.randint(7, 9),
                                    'dropout_p' : ss.uniform(.4, .1),
                                    'seed': ss.randint(0, 10000)})),


}


## use the same debugging set for all possibilities
#for p in itertools.product(['mort_icu', 'los_3'], ['debug'], [10]):
#    expm_hyperparameter_set[p] = expm_hyperparameter_set[('mort_icu', 'debug', 10)]

## use the same debugging set for all possibilities
for p in itertools.product(['mort_icu', 'los_3'], ['first'], [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10]):
    expm_hyperparameter_set[p] = expm_hyperparameter_set[('mort_icu', 'first', 10)]

