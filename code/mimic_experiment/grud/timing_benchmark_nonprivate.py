import pandas as pd
import numpy as np
import torch, logging, warnings, sys, itertools
from pathlib import Path 
from timeit import default_timer as timer
from torch.multiprocessing import Pool, cpu_count, set_start_method, current_process

sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from regression_models import LogisticRegression
from config import * 
from experiment_utils import *
from mmd_grud_utils import *
from hyperparam_sets import non_priv_hyperparameter_set, get_h_grud_nonpriv

# GPU setup 
n_procs = 15 # if using gpus, this will be per gpu
devices = {"proper_gpus": [0,1,2], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
# must be imported after gpu is set up
import torch.utils.benchmark as benchmark

##### HYPERPARAMETER SEARCH SET UP

# if use_full_train = True, the dev set will be added to train, and we will 
# evaluate on the official test set (don't do this until the very end)
use_full_train = True
## hyperparameter sets
h_pass = 'benchmarks'
hyperparameter_set = non_priv_hyperparameter_set
nbenchmarks = 10

## runs results for a set of hyperparameters on given data
def benchmark_kernel_fn(h, X_train, label_train, X_test, label_test, X_mean, run_folder, 
                        use_gpu = use_gpu): 
    """runs training and test results (on dev_data) of GRUD with given hyperparameters

    Args:
        hyperparams (dict): dict of hyperperams, see unpacking of these params below for contents
        X_train (tensor): train inputs
        label_train (tensor): train labels
        X_test (tensor): test inputs
        labels_test (tensor): test labels
        X_mean (tensor): the mean of the training data over all patients over all hours
        verbose (bool, optional): flag for printoffs. Defaults to False.

    Returns:
        dict with results and hyperparams: 
        list of training losses from training 
        list of dev/test set losses during training
        model : pythorch model trained
    """
    global nbenchmarks, device_ids
    print(f'---Starting:\t{[ (k, v) for (k,v) in h.items() ]}')
    batch_size, seed = [h[k] for k in ('batch_size','seed')]

    np.random.seed(seed)
    if use_gpu:
        device_type = get_gpu_id(device_ids)
        X_train = X_train[device_type]
        label_train = label_train[device_type]    
        X_test = X_test[device_type]
        label_test = label_test[device_type]
        X_mean = X_mean[device_type]
    else:
        device_type = 'cpu'

   
    base_params = {'X_mean': X_mean, 'input_size': X_mean.shape[2], 'device_id': device_type}
    base_params.update({k: v for k, v in h.items() 
                        if k in ('cell_size', 'hidden_size', 'batch_size', 'dropout_p')})
    if h["loss"] == "l2":
        def loss_fn(preds, targets):
            mse = torch.nn.MSELoss()
            return mse(torch.squeeze(preds), targets)
        label_type = torch.float32
        base_params.update({"apply_sigmoid" : True})
    elif h["loss"] == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        label_type = torch.float32
    if h["use_bn"] == "nobn":
        base_params.update({"use_bn": False})

    model = GRUD(**base_params)

    if use_gpu:
        model.to(device_type)

    # Make Pytorch Dataloader (batches) 
    train_dataloader = create_dataloader(X_train, label_train, batch_size=batch_size, label_type = label_type)
    test_dataloader = create_dataloader(X_test, label_test, batch_size=batch_size, label_type = label_type)

    t0 = benchmark.Timer(
        stmt='Train_Model', 
        setup='from __main__ import Train_Model',
        globals = {'model' : model, 'loss_fn' : loss_fn, 'train_dataloader' : train_dataloader, 
        'num_epochs' : h['num_epochs'], 'patience' : h['patience'], 'learning_rate' : h['learning_rate'], 
        'batch_size' : h['batch_size']}
     )

    train_time = t0.timeit(nbenchmarks)
    h['mean_train_time'] = train_time.mean
    h['median_train_time'] = train_time.median
    h['train_times'] = train_time.times
    h['nbenchmarks'] = nbenchmarks
    h['devices'] = device_ids
    h['use_gpu'] = use_gpu

    print(f'\n\n---Script run_dpsgd_grud_los_mort.py: --------- Saving results data ...')
    results_folder = Path(run_folder, f"{h['target']}_loss_{h['loss']}_{h['use_bn']}_results")
    filename = f"model_{h['model_id']}.json"
    jsonify(h, Path(results_folder, filename))
    return h#, losses[0], losses[1]#, best_model.cpu() 


if __name__ == '__main__': 
    # set up multiprocessing
    if use_gpu:
        processes = n_procs * len(device_ids) # each process will be on one gpu
    else:
        processes = n_procs
 
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)

    SEED = 1
    np.random.seed(SEED)
    
    FOLDER = Path(GRUD_BASELINE_RESULTS_FOLDER)
    FOLDER.mkdir(exist_ok=True) 
    run = 0  
    RESULTS_FOLDER = Path(FOLDER, f"{h_pass}{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, will overwrite with new results...")
        #run+= 1
        RESULTS_FOLDER = Path(FOLDER, f"{h_pass}{run}")
    RESULTS_FOLDER.mkdir(exist_ok=True)     
    losses = ['l2', 'bce']
    bns = ['bn', 'nobn']
    targets = ['mort_icu', 'los_3']

 
    for target in targets:
        print(f'\nStarting {target}\nReading in data...')
        df_train, df_test, Ys_train, Ys_test, _, _ = load_data(target, level = 2, normalization = 'z', use_full = use_full_train)

        ### get labels with correct target
        Ys_train = Ys_train[target]
        Ys_test = Ys_test[target]
        X_train = torch.from_numpy(to_3D_tensor(df_train).astype(np.float32))
        X_test = torch.from_numpy(to_3D_tensor(df_test).astype(np.float32))
        label_train = torch.from_numpy(Ys_train.values.astype(np.float32))
        label_test = torch.from_numpy(Ys_test.values.astype(np.float32))

        # take the mean over all patients over all 24 hours
        idx = pd.IndexSlice
        num_hours = 24
        x =  (df_train.loc[:, pd.IndexSlice[:, 'mean']] * np.where((df_train.loc[:, pd.IndexSlice[:, 'mask']] == 1).values, 1, np.NaN)).mean()
        X_mean = torch.Tensor( np.array([x.values] *num_hours )).unsqueeze(0)

        if use_gpu:
            X_train = dict(((gpu_id, X_train.to(gpu_id)) for gpu_id in device_ids))
            X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))
            X_mean = dict(((gpu_id, X_mean.to(gpu_id)) for gpu_id in device_ids))
            label_train = dict(((gpu_id, label_train.to(gpu_id)) for gpu_id in device_ids))
            label_test = dict(((gpu_id, label_test.to(gpu_id)) for gpu_id in device_ids))

        # we will test on bce and l2 loss with and without batchnorm
        all_hyperparams = [get_h_grud_nonpriv(target, h_pass, loss, bn, run, 
                                              hyperparameter_set, use_full_train, use_gpu, RESULTS_FOLDER)
                            for (loss, bn) in itertools.product(losses, bns)]
                
        args_list = [(h, X_train, label_train, X_test, label_test, X_mean, RESULTS_FOLDER) 
                    for h_list in all_hyperparams for h in h_list]

        # run the multiprocesing for l2 loss: 
        start = timer()
        print(f'------\n\nstarting training for {target} parallelized with {processes} cores ...\n\n----')
        
        records = pool.starmap(benchmark_kernel_fn, args_list)# should return a list of the outputs 

        end = timer()
        
        print(f"\n\n---Runs for target = {target} done! \n\t {(end-start)/60} minutes to finish")
        print(f'Results saved in {RESULTS_FOLDER}')

        # save losses so we can see if training was wonky:
        print(f"\n\n---- Printing results for {target}---\n")
        df = pd.DataFrame.from_records(records)
        print(f'\n---- Results for {target} ------')
        print(df)
        print('\n-------------\n')
