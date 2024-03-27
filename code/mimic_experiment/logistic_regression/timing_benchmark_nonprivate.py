## see Pytorch Timer source code.  
## https://github.com/pytorch/pytorch/blob/main/torch/utils/benchmark/utils/timer.py
## https://github.com/pytorch/pytorch/blob/main/torch/utils/benchmark/utils/common.py#L74 

import numpy as np
import sys
from pathlib import Path
from torch.multiprocessing import Pool, set_start_method#, cpu_count, current_process
sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from config import *
from experiment_utils import *
from hyperparam_sets import get_h_LR_nonpriv, non_priv_hyperparameter_set as  hyperparameter_set

# GPU setup 
devices = {"proper_gpus": [0,1,2], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
##### HYPERPARAMETER SEARCH SET UP ## hyperparameter pass used for folder set up and hyperparmeters to use: 
h_pass = "benchmarks" 
use_full_train = True # if use_full_train = True, the dev set will be added to train, and we will 
nbenchmarks = 10

## for multiprocessing and using gpus
proc_per_gpu = 15
processes = proc_per_gpu * len(device_ids) # each process will be on one gpu


def benchmark_kernel_fn(h, train_data, X_test, nbenchmarks,
                        run_folder, use_gpu = use_gpu, device_ids = device_ids,
                        gpu_names = gpu_names): 
    """timing benchmark results function 
    Args:
        hyperparams (dict): dict of hyperperams, see unpacking of these params below for contents
        train_data (dict): dictionary of form {gpu_id/"cpu": training dataset maker (from load_data function)}
        X_test (tensor): test inputs
        n_benchmarks (int): number of iterations
        run_folder (Path or string): folder for results
        use_gpu (Bool): True for gpu
        gpu_names (list): output of gpu setup function
    Returns:
        h: updated hyperparm dict with results and hyperparams: 
    """
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
    reduction = h['loss_reduction']

    model = LogisticRegression(X_test.shape[1], 1)
    if use_gpu: model.to(device_id)
    
    # Make Pytorch Dataloader (batches) 
    train_dataloader = DataLoader(train_data, batch_size, shuffle = False)

    t0 = benchmark.Timer(
        stmt='nonpriv_train', 
        setup='from __main__ import nonpriv_train',
        globals = {'model' : model, 'loss_fn' : loss_fn, 'train_dataloader' : train_dataloader, 
        'num_epochs' : h['num_epochs'], 'patience' : h['patience'], 'learning_rate' : h['learning_rate'], 
        'lam' : h['lam'], 'momentum' : h['momentum'],
        'verbose' : False, 'reduction' : reduction}
     )

    train_time = t0.timeit(nbenchmarks)
    h['mean_train_time'] = train_time.mean 
    h['median_train_time'] = train_time.median ## doesn't seem to give median. 
    h['train_times'] = train_time.times ## oddly this gives the mean, see common.Measurement source code
    h["raw_times"] = train_time.raw_times
    h['nbenchmarks'] = nbenchmarks
    h['devices'] = gpu_names

    print(f'\n\n---Kernel from timing_benchmark_nonprivate.py: --------- Saving results data ...')
    results_folder = Path(run_folder, f"{h['target']}_loss_{h['loss']}_results")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = f"model_{h['model_id']}.json"
    jsonify(h, Path(results_folder, filename))
    return h

if __name__ == '__main__': 
    # set up multiprocessing
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)
    ## seed
    SEED = 1
    np.random.seed(SEED)

    ## find next folder number and make a new folder:     
    FOLDER = Path(LR_BASELINE_RESULTS_FOLDER)
    run = 0
    benchmark_folder = Path(LR_BASELINE_RESULTS_FOLDER, "timing_benchmark")
    RESULTS_FOLDER = Path(benchmark_folder, f"{h_pass}{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, output will overwrite previous results...")
        # run+= 1
        # RESULTS_FOLDER = Path(benchmark_folder, f"{h_pass}{run}")
    RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)     

    for target in targets: # target = 'mort_icu'
        _, _, _, _, train_data, test_data = load_data(target, level = 3, normalization = 'z',
                                                            use_full = use_full_train)
        X_test = test_data.X
        label_test = test_data.y
        
        if use_gpu: 
            train_data = dict(((gpu_id, train_data.to(gpu_id)) for gpu_id in device_ids))
            X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))

        for loss in losses:
            ## get the hyperparameters corresponding to this target and pass (defined above)                    
            h_list = get_h_LR_nonpriv(target,  h_pass, loss, run, hyperparameter_set, use_full_train, use_gpu)
    
    ### To execute only one run, no multiprocessing
            # h = h_list[0]
            # benchmark_kernel_fn(h, train_data, X_test, nbenchmarks, RESULTS_FOLDER)
    #### comment above/uncomment below for multiprocessing, and vice versa
        
            args_list = [(h, train_data, X_test, nbenchmarks, RESULTS_FOLDER) 
                            for h in h_list]
            print(f'------\n\nstarting timing benchmark training for {target, loss} parallelized with {processes} cores ...\n\n----')
            records = pool.starmap(benchmark_kernel_fn, args_list)# should return a list of the outputs 
            print(f"\n\n---Runs for target = {target, loss} done! \n")
            print(f'Results saved in {RESULTS_FOLDER}')
            print(f"\n\n---- Printing results for {target, loss}---\n")
            df = pd.DataFrame.from_records(records)
            print(df)
            print('\n-------------\n')




