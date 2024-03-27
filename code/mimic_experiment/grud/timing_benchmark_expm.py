# %% import needed modules:  
import logging, copy, warnings, torch, sys, itertools, os
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from pathlib import Path 
from torch.multiprocessing import Pool, cpu_count, set_start_method, current_process
from multiprocessing.pool import ThreadPool
from itertools import chain
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.append(Path(".", "code", "mimic_experiment").as_posix())
sys.path.append(Path(".", "code").as_posix())
from experiment_utils import *

## GPU setup 
n_procs = 1 # if using gpus, this will be per gpu
devices = {"proper_gpus": [0,1,2], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
# must be imported after gpu set up
import torch.utils.benchmark as benchmark

from data_utils import *
# from regression_models import LogisticRegression
from normalizing_flows import * 
from config import * 
from expm_grud_utils import *
from hyperparam_sets import expm_hyperparameter_set, get_h_grud_expm

##### HYPERPARAMETER SEARCH SET UP
use_full_train = True
h_pass = 'benchmarks'
# gives number of hyperparameters to try and the space to search over
hyperparameter_set = expm_hyperparameter_set
nbenchmarks = 10

def benchmark_kernel_fn(h, X_train, label_train, X_test, label_test, X_mean, run_folder, verbose = False): 
    """
    Args:
        h (dict): hyperparameter dict
        X_1 (tensor): the training data with a row of ones at the end
        y_train (tensor): the training labels
        X_test (tensor): the test data
        y_test (tensor): the test labels
        metadata_folder (string or pathlib.Path): where to store loss outputs
        results_folder (string or pathlib.Path): where to store hyperparameter dict w/ AUC results added 
        verbose (bool, optional): True for printoffs during training 
    Returns:
        dict: h w/ auc key/value added 
    """
    global nbenchmarks, use_gpu, device_ids
    print(f'\n---Starting parameters:\t{[ (k, v) for (k,v) in h.items() ]}\n')

    ## unpack hyperparameter dict
    seed = h['seed']
    epsilon = h['epsilon']
    n_flows = h['n_flows']
    epochs = h['epochs']
    learning_rate = h['learning_rate']
    momentum = h['momentum']
    lam = h['lam']
    batch_size = h['batch_size']
    data_batch_size = h['data_batch_size']
    m = h['m']
    hh = h['hh']
    s = h['s']
    patience = h['patience']
    sample_std = h['sample_std']
    target = h['target']

    # get paths
    metadata_folder = Path(run_folder, f'{target}_epsilon_{epsilon}_training_losses')
    results_folder = Path(run_folder, f'{target}_epsilon_{epsilon}_results')

    if use_gpu:
        device_type = get_gpu_id(device_ids)
    else:
        device_type = 'cpu'


    ## Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_wrapper = create_grud_model(h, X_mean, data_batch_size, device_type, use_gpu)

    train_dataloader, test_dataloader = get_dataloaders(X_train, label_train, X_test, 
                                                        label_test, data_batch_size)
       ## setup Sylvester NF
    z_size = model_wrapper.nparams
    print("Number of params to sample: ", z_size)

    nf_model = NormalizingFlowSylvester(z_size, m = m, hh = hh, n_flows=n_flows) # number of layers/flows = n_flows

    # push model to gpu
    if use_gpu:
        print("Putting models on gpu...")
        nf_model.to(device_type)
        model_wrapper.to(device_type)

        
    # set up optimizer
    opt = torch.optim.RMSprop(
        params = nf_model.parameters(),
        lr = learning_rate,
        momentum = momentum
    )
    scheduler = ReduceLROnPlateau(opt, 'min', patience=patience, verbose = True) 


    t0 = benchmark.Timer(
        stmt='train', 
        setup='from __main__ import train',
        globals = {'model' : nf_model, 'model_wrapper' : model_wrapper, 
                   'train_dataloader': train_dataloader, 'opt' : opt, 'scheduler' : 'scheduler',
                   'epochs' : epochs, 's': s, 'lam': lam, 'z_size': z_size, 'sample_std' : sample_std,
                    'batch_size' : batch_size, 'device_type' : device_type, 
                    'verbose' : verbose})

    train_time = t0.timeit(nbenchmarks)
    print(train_time)
    h['mean_train_time'] = train_time.mean
    h['median_train_time'] = train_time.median
    h['train_times'] = train_time.times
    h['nbenchmarks'] = nbenchmarks
    h['devices'] = device_ids


   # %% now get AUC stats for parameters sampled from this trained NF
    print(f'\n\n---Script run_expm_los_mort_parallelized_gpu.py: ------Saving results data ...')
    filename = f"results_epsilon_{h['epsilon']}_m_{h['m']}_hh_{h['hh']}_epochs_{h['epochs']}_lr_{h['learning_rate']}_momentum_{h['momentum']}_lam_{h['lam']}_batch_size_{h['batch_size']}_seed_{h['seed']}.json"
    jsonify(h, Path(results_folder, filename))

    ## print results    
    return h

if __name__ == '__main__': 
    epsilons = [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] #epsilons to test over
    targets = ['mort_icu', 'los_3']
    start = timer()
    if use_gpu:
        processes = n_procs * len(device_ids) # each process will be on one gpu
    else:
        processes = n_procs
 
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)
       
    FOLDER = Path(GRUD_EXPM_RESULTS_FOLDER) 
    FOLDER.mkdir(exist_ok=True) 
    run = 0 
    run_folder = Path(FOLDER, f"{h_pass}{run}")
    if run_folder.exists(): 
        print(f"RESULTS_FOLDER {run_folder} already exists, will overwrite with new results...")
    run_folder.mkdir(exist_ok=True)     

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
        num_hours = 24
        x =  (df_train.loc[:, pd.IndexSlice[:, 'mean']] * np.where((df_train.loc[:, pd.IndexSlice[:, 'mask']] == 1).values, 1, np.NaN)).mean()
        computed_X_mean = torch.Tensor( np.array([x.values] *num_hours )).unsqueeze(0)
        # check to make sure the mean is 0 (i.e. data is normalized)
        X_mean = torch.zeros(computed_X_mean.shape)
        assert (abs(computed_X_mean) < 1e-10).all() , "Computed X_mean is not 0. Ensure data is normalized to have mean 0"

        print(f'\n---Script run_expm_los_mort_parallelized_gpu.py: Starting run = {run}, target = {target}---')

        ## Set seed for reproducibility
        SEED = 0
        np.random.seed(SEED)

        all_hyperparams = [get_h_grud_expm(target, h_pass, epsilon, use_gpu, run, 
                                           hyperparameter_set, use_full_train, run_folder)
                           for epsilon in epsilons]
                
        args_list = [(h, X_train, label_train, X_test, label_test, X_mean, run_folder) 
                    for h_list in all_hyperparams for h in h_list]

        # run the multiprocesing 
        records = pool.starmap(benchmark_kernel_fn, args_list)# should return a list of the outputs 

        df = pd.DataFrame.from_records(records)
        print(df)

    end = timer()
    print(f'\n This took {(end-start)/60} minutes')






