import numpy as np
import pandas as pd
import sys
from pathlib import Path
from torch.multiprocessing import Pool, set_start_method#, cpu_count, current_process
sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from config import *
from experiment_utils import *
from hyperparam_sets import get_h_LR_dpsgd, dpsgd_hyperparameter_set as  hyperparameter_set

# GPU setup 
devices = {"proper_gpus": [2], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
##### HYPERPARAMETER SEARCH SET UP ## hyperparameter pass used for folder set up and hyperparmeters to use: 
h_pass = "benchmarks"
use_full_train = True # if use_full_train = True, the dev set will be added to train, and we will 
nbenchmarks = 10


def benchmark_kernel_fn(h, train_data, X_test, run_folder, nbenchmarks =nbenchmarks): 
    """timing benchmark results function 
    Args:
        hyperparams (dict): dict of hyperperams, see unpacking of these params below for contents
        train_data (dict): dictionary of form {gpu_id/"cpu": training dataset maker (from load_data function)}
        X_test (tensor): test inputs
        run_folder (Path or string): folder for results
        n_benchmarks (int): number of iterations
        use_gpu (Bool): True for gpu
        gpu_names (list): output of gpu setup function
    Returns:
        h: updated hyperparm dict with results and hyperparams: 
    """
    global use_gpu, device_ids, gpu_names
    import torch.utils.benchmark as benchmark

    print(f'---Starting:\t{[ (k, v) for (k,v) in h.items() ]}')
    batch_size = int(h['batch_size'])
    seed = h['seed']
    np.random.seed(seed)
    if use_gpu:
        device_id = get_gpu_id(device_ids)
        train_data = train_data[device_id]
        X_test = X_test[device_id]
    else:
        device_id = 'cpu'

    if h["loss"] == "l2":
        loss_fn = loss_l2
    elif h["loss"] == "bce":
        loss_fn = loss_bce
    # reduction = h['loss_reduction']

    h['nbenchmarks'] = nbenchmarks
    h['devices'] = gpu_names

    model = LogisticRegression(X_test.shape[1], 1)
    if use_gpu: model.to(device_id)
    
    # Make Pytorch Dataloader (batches) 
    train_dataloader = DataLoader(train_data, batch_size, shuffle = False)
    sample_rate = 1/len(train_dataloader) # already incorporates batchsize

    # compute nm, so we can store it: 
    noise_multiplier = bm_noise_multiplier(h, sample_rate)   
    h.update({'sample_rate': sample_rate, 
              'noise_multiplier': noise_multiplier, 
              'max_grad_norm': 1})

    # time the computation of nm for benchmarking and store it:
    nm_t = benchmark.Timer(
        stmt = 'bm_noise_multiplier',
        setup = 'from __main__ import bm_noise_multiplier',
        globals = {'h' : h, 'sample_rate' : sample_rate})             
    compute_noise_multiplier_time = nm_t.timeit(nbenchmarks)
    h['mean_compute_noise_multiplier_time'] = compute_noise_multiplier_time.mean
    h['median_compute_noise_multiplier_time'] = compute_noise_multiplier_time.median
    h['compute_noise_multiplier_times'] = compute_noise_multiplier_time.times
      
    # instantiate a new model: 
    model = LogisticRegression(X_test.shape[1], 1)
    if use_gpu: model.to(device_id)

    # in order to get total_steps and privacy accountant, train model: 
    _, privacy_engine, total_steps, _ = dpsgd_train(model, loss_fn, train_dataloader, 
        **{k: v for k, v in h.items() if k in (
            'learning_rate', 'patience', 
            'momentum', 'num_epochs', 'lam', 
            'noise_multiplier',
        )}, verbose = False)
    h['total_steps'] = total_steps

    # time the training: 
    train_t = benchmark.Timer(
        stmt = 'dpsgd_train',
        setup = 'from __main__ import dpsgd_train',
        globals = {'model': model, 'loss_fn' : loss_fn, 'train_dataloader' : train_dataloader, 
        'learning_rate' : h['learning_rate'], 'patience' : h['patience'],
        'momentum': h['momentum'], 'num_epochs' : h['num_epochs'],
        'lam' : h['lam'], 'noise_multiplier' : h['noise_multiplier'],
        'verbose' : False})
    train_time = train_t.timeit(nbenchmarks)
    h['mean_train_time'] = train_time.mean
    h['median_train_time'] = train_time.median
    h['train_times'] = train_time.times
    h["raw_times"] = train_time.raw_times

    # time the epsilon computation using the trained privacy accountant: 
    ce_t = benchmark.Timer(
        stmt = 'privacy_engine.accountant.get_epsilon',
        globals = {'privacy_engine' : privacy_engine, 'delta' : h['delta'], 'eps_error' : h['eps_error']})
    compute_epsilon_time = ce_t.timeit(nbenchmarks)
    h['mean_compute_epsilon_time'] = compute_epsilon_time.mean
    h['median_compute_epsilon_time'] = compute_epsilon_time.median
    h['compute_epsilon_times'] = compute_epsilon_time.times

    print(f'\n\n---Script timing_benchmark_dpsgd.py: --------- Saving results data ...')
    results_folder = Path(run_folder, f"{h['target']}_epsilon_{h['target_epsilon']}_loss_{h['loss']}_results")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = f"model_{h['model_id']}.json"
    jsonify(h, Path(results_folder, filename))
    return h#, losses[0], losses[1]#, best_model.cpu() 


if __name__ == '__main__': 
    ## for multiprocessing and using gpus
    proc_per_gpu = 15
    processes = proc_per_gpu * len(device_ids) # each process will be on one gpu
    # set up multiprocessing
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)
    ## seed
    SEED = 1
    np.random.seed(SEED)
    ## find next folder number and make a new folder:     
    FOLDER = Path(LR_DPSGD_RESULTS_FOLDER)
    run = 0
    benchmark_folder = Path(FOLDER, "timing_benchmark")
    RESULTS_FOLDER = Path(benchmark_folder, f"{h_pass}{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, output will overwrite previous results...")
        # run+= 1
        # RESULTS_FOLDER = Path(benchmark_folder, f"{h_pass}{run}")
    RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)     

    epsilons = [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] #1e-6, 1e-5, 

    for target in targets: # target = 'mort_icu'
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
            args_list = [(h, train_data, X_test, RESULTS_FOLDER) 
                        for h in h_list]
            print(f'------\n\nstarting DPSGD timing benchmark training for {target, loss} parallelized with {processes} cores ...\n\n----')
            records = pool.starmap(benchmark_kernel_fn, args_list)# should return a list of the outputs 
            df = pd.DataFrame.from_records(records)
            print(f"\n\n---- Printing results for {target, loss}---\n")
            print(df)
            print('\n-------------\n')
            print(f'Results saved in {RESULTS_FOLDER}')
            
