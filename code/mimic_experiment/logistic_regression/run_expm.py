import torch, sys, pandas as pd, numpy as np
from pathlib import Path 
from torch.multiprocessing import Pool, set_start_method
sys.path.append(Path(".", "code").as_posix())
sys.path.append(Path(".", "code", "mimic_experiment").as_posix())
from config import *
from experiment_utils import *
from hyperparam_sets import expm_hyperparameter_set as hyperparameter_set, get_h_LR_expm

#### GPU SET UP
n_procs = 2 # if using gpus this will be per gpu
devices = {"proper_gpus": [0,1,2], "migs":[] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
##### HYPERPARAMETER SEARCH SET UP ## hyperparameter pass used for folder set up and hyperparmeters to use: 
h_pass = 'first' # options are 'first', then 'refined' (both with use_full_train = False),  then the test run:  h_pass = 'final' and use_full_train = True
use_full_train = False # if use_full_train = True, the dev set will be added to train, and we will 
# evaluate on the official test set (don't do this until the end, i.e., h_pass = 'final')


def kernel_fn(h, X_1, y_train, X_test, y_test, run_folder, use_gpu = use_gpu, device_ids = device_ids):
    """
    Args:
        h (dict): hyperparameter dict
        X_1 (tensor): the training data with a row of ones at the end
        y_train (tensor): the training labels
        X_test (tensor): the test data
        y_test (tensor): the test labels
        run_folder (string or pathlib.Path): where to store hyperparameter dict w/ AUC results added 
        use_gpu (bool): made above by setup_devices
        device_ids (list): made above by setup_devices
        gpu_names (list): made above by setup_devices
    Returns:
        dict: h w/ auc key/value added 
    """
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
    m = h['m']
    hh = h['hh']
    s = h['s']
    patience = h['patience']
    # skip = h['skip']
    sample_std = h['sample_std']
    target = h['target']
    
    if use_gpu:
        device_id = get_gpu_id(device_ids)
        X_1 = X_1[device_id]
        y_train = y_train[device_id]   
        yl = torch.stack([ y_train ]*batch_size, 1) # make labels for every y_pred value
        X_test = X_test[device_id]
        y_test = y_test[device_id]
    
    ## Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## setup Sylvester NF
    z_size = X_1.shape[1]
    model = NormalizingFlowSylvester(z_size, m = m, hh = hh, n_flows=n_flows) # number of layers/flows = n_flows

    # push model to gpu
    if use_gpu: model.to(device_id)

    nf_train(model, X_1, yl, epsilon, s, lam, z_size, sample_std, learning_rate, 
                momentum, epochs, batch_size, patience, device_id)

    # eval: 
    samples = random_normal_samples(1000, z_size, sample_std, device = device_id) # l by X.shape[0] + 1 tensor, {u_i: i = 1,..,l} sampled from base. 
    transformed_samples, _ = model.forward(samples) # l by X.shape[0] + 1 tensor, run thru NF 
    betas = transformed_samples.detach().squeeze()
    h['aucs'] = [get_auc(b, X_test, y_test) for b in betas] # puts AUC for each sampled pararamter set into dict h
    x = pd.Series(h['aucs'])
    h['auc_ave']= x.mean()  # stores mean AUC across all samples 

    # write results: 
    results_folder = Path(run_folder, f'{target}_epsilon_{epsilon}_results')
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = f"model_{h['model_id']}.json"        
    path = Path(results_folder, filename)
    print(f'\n\n---Kernel from run_expm.py: ------Saving results data to {path}...')
    jsonify(h, path)
    ## print results    
    print(f'---Finished: AUC mean is {x.mean()}')
    return h


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
    SEED = 0
    np.random.seed(SEED)
    ## data set up: 
    data_norm = 'z'
    epsilons = [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] #1e-6, 1e-5, 
          
    ## find next folder number and make a new folder:     
    FOLDER = Path(LR_EXPM_RESULTS_FOLDER) 
    FOLDER.mkdir(exist_ok=True) 
    run = 0
    RESULTS_FOLDER = Path(FOLDER, f"{h_pass}{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, will write new results to existing folder...")
        #run+= 1
        # RESULTS_FOLDER = Path(FOLDER, f"{h_pass}{run}")
    RESULTS_FOLDER.mkdir(exist_ok=True)     

    for target in targets:
     ## read in data. if on gpu, it will read it as a tensor so it can be pushed to gpu
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

        print(f'\n---Script run_expm.py: Starting run = {run}, target = {target}---')
        h_list = get_h_LR_expm(target, h_pass, epsilons, run, 
                               hyperparameter_set=hyperparameter_set, 
                               use_full_train=use_full_train, use_gpu = use_gpu, 
                               data_norm = data_norm)
        args_list = [(h, X_1, y_train, X_test, y_test, RESULTS_FOLDER) 
                    for h in h_list]# list of arg tuples for kernel: 

        # run the multiprocesing 
        outputs = pool.starmap(kernel_fn, args_list)# should return a list of the outputs 
        df = pd.DataFrame.from_records(outputs)
        print(f"\n\n---Runs for {target} done!")
        print(f'Results saved in {RESULTS_FOLDER}')
        print(f"\n\n---- Printing results for {target}---\n")
        print(df)
        print('\n-------------\n')
