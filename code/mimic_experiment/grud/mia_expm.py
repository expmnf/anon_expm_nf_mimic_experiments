# %% import needed modules:  
import logging, copy, warnings, torch, sys, itertools, os, scipy
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
devices = {"proper_gpus": [0,1,2,3], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)

# Membership Inference hyperparameters
conf_type = "stable" # can be "stable" or "hinge"
total_target_models = 50

from data_utils import *
# from regression_models import LogisticRegression
from normalizing_flows import * 
from config import * 
from expm_grud_utils import *
from hyperparam_sets import expm_hyperparameter_set, get_h_grud_expm



# requires shadow models have been trained before running this
def kernel_fn(h, mia_data_dict, all_gpu_X_train, all_gpu_label_train, all_gpu_X_test, all_gpu_label_test, 
              mean_df, num_hours, run_folder, verbose = False):

    """
    Args:
        h (dict): hyperparameter dict
        mia_data_dict (dict): shadow confidences
        X_1 (tensor): the training data with a row of ones at the end
        y_train (tensor): the training labels
        X_test (tensor): the test data
        y_test (tensor): the test labels
        metadata_folder (string or pathlib.Path): where to store loss outputs
        X_mean (tensor): the mean of the training data over all patients over all hours
        verbose (bool, optional): flag for printoffs. Defaults to False.
        results_folder (string or pathlib.Path): where to store hyperparameter dict w/ AUC results added 
        verbose (bool, optional): True for printoffs during training 

    Returns:
        dict: h w/ auc key/value added 
        dict: confidence, audit score and attack score per target model per data point
    """
    global use_gpu, device_ids
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
    #skip = h['skip']
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

    if use_gpu:
        device_type = get_gpu_id(device_ids)
        all_X_train = all_gpu_X_train[device_type]
        all_label_train = all_gpu_label_train[device_type]    
        X_test = all_gpu_X_test[device_type]
        label_test = all_gpu_label_test[device_type]
    else:
        device_type = 'cpu'
        all_X_train = all_gpu_X_train
        all_label_train = all_gpu_label_train
        X_test = all_gpu_X_test
        label_test = all_gpu_label_test

    target_all_inds = torch.randperm(len(all_label_train))
    # during training of grud we can only use a multiple of the batchsize
    # so we will use up to half the data points and the rest will be out
    N_in = ((len(target_all_inds)//2)//batch_size)*batch_size
    target_train_inds = target_all_inds[0:N_in]
    target_X_train = all_X_train[target_train_inds]
    target_label_train = all_label_train[target_train_inds]
    target_mean_df_ids = mean_df.index.get_level_values('subject_id').unique()[target_all_inds]
    target_x = mean_df.loc[target_mean_df_ids].mean()

    computed_X_mean = torch.Tensor( np.array([target_x.values] *num_hours )).unsqueeze(0)
    target_X_mean = torch.zeros(computed_X_mean.shape)
    assert (abs(computed_X_mean) < 1e-10).all() , "Computed X_mean is not 0. Ensure data is normalized to have mean 0"

    # grud drops the extra data points, but we need them so we will make another batch at the end using the first
    # values and just not use them...
    diff = len(target_all_inds) - len(target_all_inds)//data_batch_size*data_batch_size 
    if diff != 0:
        to_add = data_batch_size - diff
        bigger_X_train = torch.concat((all_X_train, all_X_train[range(0,to_add)]))
        bigger_label_train = torch.concat((all_label_train, all_label_train[range(0,to_add)]))
        all_dataloader = create_dataloader(bigger_X_train, bigger_label_train, batch_size=data_batch_size)
    else:
        all_dataloader = create_dataloader(all_X_train, all_label_train, batch_size=data_batch_size)


    # Make Pytorch Dataloader (batches) 
    target_train_dataloader = create_dataloader(target_X_train, target_label_train, batch_size=batch_size)

    model_wrapper = create_grud_model(h, target_X_mean, data_batch_size, device_type, use_gpu)

    train_dataloader, test_dataloader = get_dataloaders(target_X_train, target_label_train, X_test, 
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
     
    # get a target model params:
    param_sample = random_normal_samples(1, z_size, sample_std, device = device_type)
    b, _ = nf_model.forward(param_sample)
    # auditing
    loss_fn = torch.nn.MSELoss(reduction = 'none')
    score_before = score(model_wrapper, b, loss_fn, all_dataloader, device_type, use_gpu)
       
    # set up optimizer
    opt = torch.optim.RMSprop(
        params = nf_model.parameters(),
        lr = learning_rate,
        momentum = momentum
    )

    scheduler = ReduceLROnPlateau(opt, 'min', patience=patience, verbose = True) 
    nf_model = train(nf_model, model_wrapper, train_dataloader, opt, scheduler, epochs, 
                     batch_size, z_size, sample_std, use_gpu,
                     lam, epsilon, s, device_type, verbose)

    h = evaluate_nf(nf_model, model_wrapper, h, test_dataloader, z_size, sample_std, device_type, use_gpu)

    ######## Target model training complete
    # get a target model params:
    param_sample = random_normal_samples(1, z_size, sample_std, device = device_type) # l by X.shape[0] + 1 tensor, {u_i: i = 1,..,l} sampled from base. 
    b, _ = nf_model.forward(param_sample)

    # auditing
    score_after = score(model_wrapper, b, loss_fn, all_dataloader, device_type, use_gpu)
    scores = (score_before - score_after).cpu().numpy()

    ######## Train shadow models following 
    # Carlini et al. - 2022 - Membership Inference Attacks From First Principles
    
    # this dictionary will hold membership inference attack information 
    target_data_dict = {}
    # for each data point in our training set (half were used to train the 
    # target model)
    # compute confidences for those included in training and those that weren't
    if conf_type == "stable":
        confs_obs = logit_confs_stable_expm(model_wrapper, b.detach(), all_dataloader, device_type, use_gpu).detach().cpu().numpy()
    elif conf_type == "hinge":
        confs_obs = logit_confs_grud_hinge_expm(model_wrapper, b.detach(), all_dataloader, device_type, use_gpu).detach().cpu().numpy()
    
    for ind in range(0, len(target_all_inds)):
        i = str(ind)
        target_data_dict[i] = {}
        target_data_dict[i]["target_model_member"] = {h['model_id']: ind in target_train_inds}
        target_data_dict[i]["conf_obs"] = {h['model_id']: confs_obs[ind]}
        pr_in = -scipy.stats.norm.logpdf(confs_obs[ind], mia_data_dict[i]["confs_in_mean"], 
                                                       mia_data_dict[i]["confs_in_std"]+1e-30)
        pr_out = -scipy.stats.norm.logpdf(confs_obs[ind], mia_data_dict[i]["confs_out_mean"], 
                                                       mia_data_dict[i]["confs_out_std"]+1e-30)
        target_data_dict[i]["attack_score"] = {h['model_id']: pr_in-pr_out}
        target_data_dict[i]["audit_score"] = {h['model_id']: scores[ind]}

    results_folder = Path(run_folder, f"expm_{total_target_models}", "hyperparams", f"{h['target']}_epsilon_{h['epsilon']}_results")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = f"model_{h['model_id']}.json"
    jsonify(h, Path(results_folder, filename))
    ## print results    
    print(f'\n---Script run_expm_los_mort_parallelized_gpu.py: ---Finished:\t{[ (k, v) for (k,v) in h.items() if k not in ["aucs", "betas"] ]}\n------ AUC mean is {h["auc_ave"]}')
    return h, target_data_dict


if __name__ == '__main__': 
    epsilons = [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] #epsilons to test over
    targets = ['mort_icu', 'los_3']
    bn = "nobn"
    SEED = 1
    start = timer()
    if use_gpu:
        processes = n_procs * len(device_ids) # each process will be on one gpu
    else:
        processes = n_procs
 
    use_full_train = False #True
    ## hyperparameter sets
    h_pass = 'final'
    N_shadow_models = 1000
    # gives number of hyperparameters to try and the space to search over
    hyperparameter_set = expm_hyperparameter_set

    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)
       
    ## find next folder number and make a new folder:     
    FOLDER = Path(GRUD_MIA_RESULTS_FOLDER)
    FOLDER.mkdir(exist_ok=True) 
    ## find next folder number and make a new folder:     
    run = f"{conf_type}_{N_shadow_models}"
    run_folder = Path(FOLDER, f"run_{run}")
    if run_folder.exists(): 
        print(f"RESULTS_FOLDER {run_folder} already exists, will overwrite with new results...")
    run_folder.mkdir(exist_ok=True)     

    for target in targets:
        for epsilon in epsilons:
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
            mean_df =  (df_train.loc[:, pd.IndexSlice[:, 'mean']] * np.where((df_train.loc[:, pd.IndexSlice[:, 'mask']] == 1).values, 1, np.NaN))

            if use_gpu:
                X_train = dict(((gpu_id, X_train.to(gpu_id)) for gpu_id in device_ids))
                X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))
                label_train = dict(((gpu_id, label_train.to(gpu_id)) for gpu_id in device_ids))
                label_test = dict(((gpu_id, label_test.to(gpu_id)) for gpu_id in device_ids))
                
            print("------\n\n Loading membership inference shadow model results ... \n\n------")
            mia_path = Path(run_folder, "shadow_models", f"{target}_loss_bce_{bn}_results", "mia_shadow_confs.json")
            mia_data_dict = pd.DataFrame.from_records(unjsonify(mia_path)).to_dict()

            print(f'\n---Script run_expm_los_mort_parallelized_gpu.py: Starting run = {run}, target = {target}---')

            ## Set seed for reproducibility
            SEED = 0
            np.random.seed(SEED)

            all_hyperparams = [get_h_grud_expm(target, h_pass, epsilon, use_gpu, "0",
                                               hyperparameter_set, use_full_train, 
                                               run_folder = GRUD_EXPM_RESULTS_FOLDER,
                                               final_n = total_target_models)]
    #                           for epsilon in epsilons]

            args_list = [(h, mia_data_dict, X_train, label_train, X_test, label_test, 
                         mean_df, num_hours, run_folder) 
                        for h_list in all_hyperparams for h in h_list]

            # run the multiprocesing 
            records = pool.starmap(kernel_fn, args_list)# should return a list of the outputs 

            mia_data_dicts = [r[1] for r in records]
            mia_df = pd.DataFrame.from_records(mia_data_dicts)
            # we won't save the shadow model confidences, we can get those from the shadow models.
            def combine_confs(lst):
                out = {"target_model_member": {},
                       "conf_obs": {},
                       "audit_score": {},
                        "attack_score": {}}
                for d in lst:
                    out["target_model_member"] = out["target_model_member"] | d["target_model_member"]
                    out["conf_obs"] = out["conf_obs"] | d["conf_obs"]
                    out["attack_score"] = out["attack_score"] | d["attack_score"]
                    out["audit_score"] = out["audit_score"] | d["audit_score"]
                return out
                   
            final_mia_data_dict = mia_df.apply(combine_confs).to_dict()
            results_folder = Path(run_folder, f"expm_{total_target_models}", f"{target}_loss_results")
            results_folder.mkdir(exist_ok=True, parents=True)
            mia_filename = f"mia_expm_epsilon_{epsilon}.json"
            mia_path = Path(results_folder, mia_filename)
            
            print(f'Results saved in {mia_path}')
            jsonify(final_mia_data_dict, mia_path)

            df = pd.DataFrame.from_records(records)
            print(df)

        end = timer()
        print(f'\n This took {(end-start)/60} minutes')






