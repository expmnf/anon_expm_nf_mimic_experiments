import sys
import pandas as pd
import numpy as np
from pathlib import Path
from torch.multiprocessing import Pool, set_start_method#, cpu_count, current_process
sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from regression_models import LogisticRegression
from config import * 
from experiment_utils import *
from hyperparam_sets import dpsgd_hyperparameter_set as hyperparameter_set, get_h_LR_dpsgd

# GPU setup 
n_procs = 2 # if using gpus, this will be per gpu
devices = {"proper_gpus": [0,1,2], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
##### HYPERPARAMETER SEARCH SET UP ## hyperparameter pass used for folder set up and hyperparmeters to use: 
h_pass = 'first' # options are 'first', then 'refined' (both with use_full_train = False),  # then the test run:  h_pass = 'final' and use_full_train = True
use_full_train = False # if use_full_train = True, the dev set will be added to train, and we will 
# evaluate on the official test set (don't do this until the end, i.e., h_pass = 'final')

## runs results for a set of hyperparameters on given data
def kernel_fn(h, train_data, X_test, y_test, run_folder): 
    """runs training and test results (on dev_data) of GRUD with given hyperparameters
    Args:
        hyperparams (dict): dict of hyperperams, see unpacking of these params below for contents
        train_data (dict): dictionary of form {gpu_id/"cpu": training dataset maker (from load_data function)}
        X_test (tensor): test inputs
        labels_test (tensor): test labels
        run_folder (Path or string): folder for results
    Returns:
        h: updated hyperparm dict with results and hyperparams: 
    """
    global use_gpu, device_ids
    print(f'---Starting:\t{[ (k, v) for (k,v) in h.items() ]}')
    batch_size = int(h['batch_size'])
    seed = h['seed']
    np.random.seed(seed)
    if use_gpu:
        device_type = get_gpu_id(device_ids)
        train_data = train_data[device_type]
        X_test = X_test[device_type]
    else:
        device_type = 'cpu'

    # Make Pytorch Dataloader (batches) 
    train_dataloader = DataLoader(train_data, batch_size, shuffle = True)
   
    if h["loss"] == "l2":
        loss_fn = loss_l2
    elif h["loss"] == "bce":
        loss_fn = loss_bce

    sample_rate = 1/len(train_dataloader) # already incorporates batchsize
    noise_multiplier = bm_noise_multiplier(h, sample_rate)

    h.update({'sample_rate': sample_rate, 
              'noise_multiplier': noise_multiplier, 
              'max_grad_norm': 1})
 
    model = LogisticRegression(X_test.shape[1], 1)

    if use_gpu: model.to(device_type)
        
    best_model, privacy_engine, total_steps, _ = dpsgd_train(model, loss_fn, train_dataloader, 
        **{k: v for k, v in h.items() if k in (
            'learning_rate', 'patience', 
            'momentum', 'num_epochs', 'lam', 
            'noise_multiplier',
        )}, verbose = False)
    
    h['total_steps'] = total_steps
    try:
        h['prv_epsilon'] = privacy_engine.accountant.get_epsilon(h['delta'], eps_error = h['eps_error'])
    except: 
        h['prv_epsilon'] = np.nan
    
    # record metrics by adding them to the  hyperparms dict
    y_score = best_model(X_test).squeeze().detach().cpu()

    h['auc'] = roc_auc_score(y_test, y_score)
    h['aps'] = average_precision_score(y_test, y_score)
    
    # get best threhold 
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresh":thresholds})
    df['dist_to_[0,1]'] = df.apply( lambda row: np.sqrt((1-row.tpr)**2 + (row.fpr)**2), axis = 1) # dist to [0,1]
    best = df[ df['dist_to_[0,1]'] == df['dist_to_[0,1]'].min()].iloc[0] # closest point on ROC to [0,1]
    thresh = best.thresh
    y_pred = (y_score>thresh) # binary predictions

    # metrics at "best" threshold
    best['accuracy'] = accuracy_score(y_test, y_pred)
    best['f1']  = f1_score(y_test, y_pred)
    best['prec'] = precision_score(y_test, y_pred)
    
    # add to results: 
    h.update(best.to_dict())
    
    print(f'\n\n---Script run_dpsgd.py: --------- Saving results data ...')
    results_folder = Path(run_folder, f"{h['target']}_epsilon_{h['target_epsilon']}_loss_{h['loss']}_results")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = f"model_{h['model_id']}.json"
    jsonify(h, Path(results_folder, filename))
    print(f"---Finished: AUC = {h['auc']}")
    return h#, losses[0], losses[1]#, best_model.cpu() 


if __name__ == '__main__': 
    # set up multiprocessing
    ## for multiprocessing and using gpus
    if use_gpu:
        processes = n_procs * len(device_ids) # each process will be on one gpu
    else:
        processes = n_procs
    
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)
    ## seed
    SEED = 1
    np.random.seed(SEED)
    ## find next folder number and make a new folder:     
    folder = Path(LR_DPSGD_RESULTS_FOLDER)
    run = 0
    RESULTS_FOLDER = Path(folder, f"{h_pass}{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, will write new results to existing folder...")
#        run+= 1
        RESULTS_FOLDER = Path(folder, f"{h_pass}{run}")
    RESULTS_FOLDER.mkdir(exist_ok=True)     

    epsilons =  [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] #epsilons to test over 
    for target in targets:
        _, _, _, _, train_data, test_data = load_data(target, level = 3, normalization = 'z',
                                                    use_full = use_full_train)
        X_test = test_data.X
        label_test = test_data.y
        if use_gpu:
            train_data = dict(((gpu_id, train_data.to(gpu_id)) for gpu_id in device_ids))
            X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))
        for loss in losses:
            h_list = get_h_LR_dpsgd(target, h_pass, loss, epsilons, run, hyperparameter_set,
                                    use_full_train, use_gpu)    
            args_list = [(h, train_data, X_test, label_test, RESULTS_FOLDER) 
                        for h in h_list]

            # run the multiprocesing for l2 loss: 
            records = pool.starmap(kernel_fn, args_list)# should return a list of the outputs 
        
            df = pd.DataFrame.from_records(records)
            print(f'\n---- Results for  {target, loss}: ')
            print(df)
            print('\n-------------\n')
            print(f'Results saved in {RESULTS_FOLDER}')

