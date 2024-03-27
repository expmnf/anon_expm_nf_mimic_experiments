import torch, logging, warnings, sys, pandas as pd, numpy as np
from pathlib import Path 
from timeit import default_timer as timer
from torch.multiprocessing import Pool,  set_start_method#, current_process
from opacus.accountants.utils import get_noise_multiplier
logging.getLogger('matplotlib.font_manager').disabled = True # turns off warnings about fonts. 
warnings.simplefilter(action='ignore', category=FutureWarning) #supress future warnings
warnings.simplefilter("ignore")

sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from config import * 
from experiment_utils import *
from mmd_grud_utils import *
from hyperparam_sets import dpsgd_hyperparameter_set, get_h_grud_dpsgd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #supress future warnings
# GPU setup 
n_procs = 15 # if using gpus, this will be per gpu
devices = {"proper_gpus": [0,1,2], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)

##### HYPERPARAMETER SEARCH SET UP
# if use_full_train = True, the dev set will be added to train, and we will 
# evaluate on the official test set (don't do this until the very end)
use_full_train = False # True
## hyperparameter sets
h_pass = 'first' ## other options are  'refined' for the second run, and 'final' 
# gives number of hyperparameters to try and the space to search over
hyperparameter_set = dpsgd_hyperparameter_set

def bm_noise_multiplier(h, sample_rate):
     try:
        noise_multiplier = get_noise_multiplier(target_epsilon = h['target_epsilon'],
                                           target_delta = h['delta'],
                                           sample_rate = sample_rate,
                                           epochs = h['num_epochs'],
                                           epsilon_tolerance = h['eps_error'],
                                           accountant = 'prv',
                                           eps_error = h['eps_error'])
     except:
        # the prv accountant is not robust to large epsilon (even epsilon = 10)
        # so we will use rdp when it fails, so the actual epsilon may be slightly off
        # see https://github.com/pytorch/opacus/issues/604
        noise_multiplier = get_noise_multiplier(target_epsilon = h['target_epsilon'],
                                                target_delta = h['delta'],
                                                sample_rate = sample_rate,
                                                epochs = h['num_epochs'],
                                                epsilon_tolerance = h['eps_error'],
                                                accountant = 'rdp')

     return noise_multiplier
 
 
def kernel_fn(h, X_train, label_train, X_test, label_test, X_mean, run_folder, verbose = False): 
    """runs training and test results (on dev_data) of dpsgd GRUD with given hyperparameters

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
        model : pythorch model trained
    """
    global use_gpu, device_ids
    print(f'---Starting:\t{[ (k, v) for (k,v) in h.items() ]}')
#    early_stop_frac, batch_size, seed = [h[k] for k in ('early_stop_frac','batch_size','seed')]
    batch_size, seed = [h[k] for k in ('batch_size','seed')]

    results_folder = Path(run_folder, f"{h['target']}_epsilon_{h['target_epsilon']}_loss_{h['loss']}_{h['use_bn']}_results")

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

    # Make Pytorch Dataloader (batches) 
    train_dataloader = create_dataloader(X_train, label_train, batch_size=batch_size)
    test_dataloader = create_dataloader(X_test, label_test, batch_size=batch_size)

    sample_rate = 1/len(train_dataloader) # already incorporates batchsize
    try:
        noise_multiplier = get_noise_multiplier(target_epsilon = h['target_epsilon'],
                                           target_delta = h['delta'],
                                           sample_rate = sample_rate,
                                           epochs = h['num_epochs'],
                                           epsilon_tolerance = h['eps_error'],
                                           accountant = 'prv',
                                           eps_error = h['eps_error'])
    except:
        # the prv accountant is not robust to large epsilon (even epsilon = 10)
        # so we will use rdp when it fails, so the actual epsilon may be slightly off
        # see https://github.com/pytorch/opacus/issues/604
        noise_multiplier = get_noise_multiplier(target_epsilon = h['target_epsilon'],
                                                target_delta = h['delta'],
                                                sample_rate = sample_rate,
                                                epochs = h['num_epochs'],
                                                epsilon_tolerance = h['eps_error'],
                                                accountant = 'rdp')



    h.update({'sample_rate': sample_rate, 
              'noise_multiplier': noise_multiplier, 
              'max_grad_norm': 1})
   
    # instantiate model:
    base_params = {'X_mean': X_mean, 'input_size': X_mean.shape[2], 'device_id': device_type}
    base_params.update({k: v for k, v in h.items() if k in ('cell_size', 'hidden_size', 'batch_size')})
    if h["loss"] == "l2":
        def loss_fn(preds, targets):
            mse = torch.nn.MSELoss()
            return mse(torch.squeeze(preds), targets)
        base_params.update({"apply_sigmoid" : True})
    elif h["loss"] == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    if h["use_bn"] == "nobn":
        base_params.update({"use_bn": False})

    model = GRUD(**base_params)
    if use_gpu:
        model.to(device_type)

    best_model, niter_per_epoch, privacy_engine = Train_Model_DPSGD(
        model, loss_fn, train_dataloader, noise_multiplier, 
        **{k: v for k, v in h.items() if k in (
            'num_epochs', 'patience', 'learning_rate', 'batch_size', 'max_grad_norm'
        )})
    # done training!
    
    # record metrics by adding them to the  hyperparms dict
    probabilities_dev, labels_dev = predict_proba(best_model, test_dataloader)
    y_score = np.concatenate(probabilities_dev)
    y_pred  = np.argmax(probabilities_dev)
    targets  = np.concatenate(labels_dev)

    h['niter_per_epoch'] = niter_per_epoch
    h['auc'] = roc_auc_score(targets, y_score)
    h['aps'] = average_precision_score(targets, y_score)
    
    # get best threhold 
    fpr, tpr, thresholds = roc_curve(targets, y_score)
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresh":thresholds})
    df['dist_to_[0,1]'] = df.apply( lambda row: np.sqrt((1-row.tpr)**2 + (row.fpr)**2), axis = 1) # dist to [0,1]
    best = df[ df['dist_to_[0,1]'] == df['dist_to_[0,1]'].min()].iloc[0] # closest point on ROC to [0,1]
    thresh = best.thresh
    y_pred = (y_score>thresh) # binary predictions

    # metrics at "best" threshold
    best['accuracy'] = accuracy_score(targets, y_pred)
    best['f1']  = f1_score(targets, y_pred)
    best['prec'] = precision_score(targets, y_pred)
    
    # add to results: 
    h.update(best.to_dict())
    # get epsilon
    h['prv_epsilon'] = privacy_engine.accountant.get_epsilon(h['delta'], eps_error = h['eps_error'])
    
    print(f'\n\n---Script run_dpsgd_grud_los_mort.py: --------- Saving results data ...')
    filename = f"model_{h['model_id']}.json"
    jsonify(h, Path(results_folder, filename))
    print(f"---Finished: AUC = {h['auc']} with epsilon = {h['prv_epsilon']}")
    
    return h


if __name__ == '__main__': 
    epsilons = [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] # epsilons to run over
    delta = 1e-5
    targets = ['mort_icu', 'los_3']
    SEED = 1
    if use_gpu:
        processes = n_procs * len(device_ids) # each process will be on one gpu
    else:
        processes = n_procs
 
    # set up multiprocessing
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)
    
    FOLDER = Path(GRUD_DPSGD_RESULTS_FOLDER)
    FOLDER.mkdir(exist_ok=True) 
    ## find next folder number and make a new folder:     
    run = 0 # increment each run to not overwrite previous results! 
    run_folder = Path(FOLDER, f"{h_pass}{run}")
    if run_folder.exists(): 
        print(f"RESULTS_FOLDER {run_folder} already exists, will overwrite with new results...")
        run_folder = Path(FOLDER, f"{h_pass}{run}")
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
        idx = pd.IndexSlice
        num_hours = 24
        x =  (df_train.loc[:, pd.IndexSlice[:, 'mean']] * np.where((df_train.loc[:, pd.IndexSlice[:, 'mask']] == 1).values, 1, np.NaN)).mean()
        computed_X_mean = torch.Tensor( np.array([x.values] *num_hours )).unsqueeze(0)
        X_mean = torch.zeros(computed_X_mean.shape)
        assert (abs(computed_X_mean) < 1e-10).all() , "Computed X_mean is not 0. Ensure data is normalized to have mean 0"

        if use_gpu:
            X_train = dict(((gpu_id, X_train.to(gpu_id)) for gpu_id in device_ids))
            X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))
            X_mean = dict(((gpu_id, X_mean.to(gpu_id)) for gpu_id in device_ids))
            label_train = dict(((gpu_id, label_train.to(gpu_id)) for gpu_id in device_ids))
            label_test = dict(((gpu_id, label_test.to(gpu_id)) for gpu_id in device_ids))
            
        np.random.seed(SEED)


        all_hyperparams = [get_h_grud_dpsgd(target, h_pass, loss, "nobn", epsilon, delta,
                                            run, hyperparameter_set, use_full_train, use_gpu, 
                                            run_folder)
                           for loss in ['l2', 'bce'] for epsilon in epsilons]
        
        args_list = [(h, X_train, label_train, X_test, label_test, X_mean, run_folder) 
                    for h_list in all_hyperparams for h in h_list]
 
        start = timer()
        print(f'------\n\nstarting training for {target} parallelized with {processes} cores ...\n\n----')
        records = pool.starmap(kernel_fn, args_list)# should return a list of the outputs 
        end = timer()
        
        print(f"\n\n---Runs for target = {target} done! \n\t {(end-start)/60} minutes to finish")
        print(f'Results saved in {Path(RESULTS_FOLDER, f"{target}_{run}.json")}')

        print(f"\n\n---- Printing results for {target}---\n")
        df = pd.DataFrame.from_records(records)
        print(f'\n---- Results for {target} ------')
        print(df)
        print('\n-------------\n')
