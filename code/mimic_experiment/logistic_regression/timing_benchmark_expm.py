import numpy as np
import sys
from pathlib import Path
from torch.multiprocessing import Pool, set_start_method#, cpu_count, current_process
sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from config import *
from experiment_utils import *
from hyperparam_sets import get_h_LR_expm, non_priv_hyperparameter_set as hyperparameter_set
# GPU setup 
devices = {"proper_gpus": [0,1], "migs":[] } # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
##### HYPERPARAMETER SEARCH SET UP ## hyperparameter pass used for folder set up and hyperparmeters to use: 
h_pass = 'benchmarks' 
use_full_train = True # if use_full_train = True, the dev set will be added to train, and we will 
nbenchmarks = 10


def benchmark_kernel_fn(h, X_1, y_train, X_test, y_test, run_folder,
                        nbenchmarks = nbenchmarks): 
    """timing benchmark results function 
    Args:
        hyperparams (dict): dict of hyperperams, see unpacking of these params below for contents
        train_data (dict): dictionary of form {gpu_id/"cpu": training dataset maker (from load_data function)}
        X_test (tensor): test inputs
        run_folder (Path or string): folder for results
        n_benchmarks (int, optional): number of iterations
    Returns:
        h: updated hyperparm dict with results and hyperparams: 
    """
    global use_gpu, device_ids, gpu_names
    import torch.utils.benchmark as benchmark

    ## unpack hyperparameter dict
    target = h["target"]
    seed = h['seed']
    epsilon = h['epsilon']
    n_flows = h['n_flows']
    epochs = h['epochs']
    learning_rate = h['learning_rate']
    momentum = h['momentum']
    lam = h['lam']
    batch_size = h['batch_size']
    m = h['m']
    hh = h['hh']
    s = h['s']
    patience = h['patience']
    sample_std = h['sample_std']
    ## Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_gpu:
        device_id = get_gpu_id(device_ids)
        X_1 = X_1[device_id]
        y_train = y_train[device_id]   
        yl = torch.stack([ y_train ]*batch_size, 1) # make labels for every y_pred value
        X_test = X_test[device_id]
        y_test = y_test[device_id]
    else:
        device_id = 'cpu'

    z_size = X_1.shape[1]
    model = NormalizingFlowSylvester(z_size, m = m, hh = hh, n_flows=n_flows) # number of layers/flows = n_flows
    if use_gpu: model.to(device_id)
    
    t0 = benchmark.Timer(
        stmt='nf_train', 
        setup='from __main__ import nf_train',
        globals = {'model' : model, 'X1' : X_1, 'yl' : yl, 'epsilon': epsilon,
                    's': s, 'lam': lam, 'z_size': z_size, 'sample_std' : sample_std,
                    'learning_rate': learning_rate, 'momentum' : momentum, 'epochs' : epochs, 
                    'batch_size' : batch_size, 'patience' : patience, 'device_type' : device_id}
     )

    train_time = t0.timeit(nbenchmarks)
    h['mean_train_time'] = train_time.mean
    h['median_train_time'] = train_time.median
    h['train_times'] = train_time.times
    h["raw_times"] = train_time.raw_times
    h['nbenchmarks'] = nbenchmarks
    h['devices'] = gpu_names
    filename = f"model_{h['model_id']}.json"

    print(f'\n\n---Kernel from timing_benchmark_expm.py: --------- Saving results data ...')
    results_folder = Path(run_folder, f'{target}_epsilon_{epsilon}_results')
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = f"model_{h['model_id']}.json"
    jsonify(h, Path(results_folder, filename))
    return h


if __name__ == '__main__': 
    # set up multiprocessing
    ## for multiprocessing and using gpus
    proc_per_gpu = 15
    processes = proc_per_gpu * len(device_ids) # each process will be on one gpu
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)
    ## seed
    SEED = 1
    np.random.seed(SEED)
    ## find next folder number and make a new folder:     
    FOLDER = Path(LR_BASELINE_RESULTS_FOLDER)
    run = 0
    benchmark_folder = Path(LR_EXPM_RESULTS_FOLDER, "timing_benchmark")
    RESULTS_FOLDER = Path(benchmark_folder, f"{h_pass}{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, output will overwrite previous results...")
        # run+= 1
        # RESULTS_FOLDER = Path(benchmark_folder, f"{h_pass}{run}")
    RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)     

    ## data set up: 
    data_norm = 'z'
    epsilons = [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] #1e-6, 1e-5, 

    for target in targets:
     ## read in data# if on gpu, it will read it as a tensor so it can be pushed to gpu
        _, _, _, _, train_data, test_data = load_data(target, level = 3, normalization = 'z',
                                                            use_full = use_full_train)
        X_train =  train_data.X
        X_1 = torch.concat([X_train, torch.ones([X_train.shape[0], 1]) ], axis = 1) # adds 1s for last column 
        y_train = train_data.y.squeeze()
        X_test = test_data.X
        y_test = test_data.y.squeeze()

        if use_gpu: # put data on gpu, keep dict of what tensors are where (to be access by child process)
            X_1 = dict(((gpu_id, X_1.to(gpu_id)) for gpu_id in device_ids))
            y_train = dict(((gpu_id, y_train.to(gpu_id)) for gpu_id in device_ids))
            X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))
            y_test = dict(((gpu_id, y_test.to(gpu_id)) for gpu_id in device_ids))
        
        print(f'\n---Script timing_benchmark_expm.py: Starting run = {run}, target = {target}---')
        h_list = get_h_LR_expm(target, h_pass, epsilons, run, 
                               hyperparameter_set=hyperparameter_set, 
                               use_full_train=use_full_train, use_gpu = use_gpu, 
                               data_norm = data_norm) 
    ### To execute only one run, no multiprocessing
            # h = h_list[0]
            # benchmark_kernel_fn(h,....)
    #### comment above/uncomment below for multiprocessing, and vice versa
        args_list = [(h, X_1, y_train, X_test, y_test, RESULTS_FOLDER) 
                    for h in h_list]# list of arg tuples for kernel:     
        
        print(f'------\n\nstarting timing benchmark training for {target} parallelized with {processes} cores ...\n\n----')
        records = pool.starmap(benchmark_kernel_fn, args_list)# should return a list of the outputs 
        print(f"\n\n---Runs for target = {target} done! \n")
        print(f'Results saved in {RESULTS_FOLDER}')
        print(f"\n\n---- Printing results for {target}---\n")
        df = pd.DataFrame.from_records(records)
        print(df)
        print('\n-------------\n')
