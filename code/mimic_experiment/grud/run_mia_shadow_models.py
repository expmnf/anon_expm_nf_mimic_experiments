import pandas as pd
import numpy as np
import torch, logging, warnings, sys, itertools
from pathlib import Path 
from timeit import default_timer as timer
from torch.multiprocessing import Pool, cpu_count, set_start_method, current_process
from matplotlib import pyplot as plt
sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from regression_models import LogisticRegression
from config import * 
from experiment_utils import *
from mmd_grud_utils import *
from hyperparam_sets import non_priv_hyperparameter_set, get_h_grud_nonpriv

# GPU setup 
n_procs = 30 # if using gpus, this will be per gpu
devices = {"proper_gpus": [0,2,3], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
conf_type = "hinge"
N_shadow_models = 1000


## runs results for a set of hyperparameters on given data
def kernel_fn(h, all_gpu_X_train, all_gpu_label_train, all_gpu_X_test, 
              all_gpu_label_test, mean_df, num_hours, seed, run_folder): 
    """runs training and test results (on dev_data) of GRUD with given hyperparameters

    Args:
        hyperparams (dict): dict of hyperperams, see unpacking of these params below for contents
        X_train (tensor): train inputs
        label_train (tensor): train labels
        X_test (tensor): test inputs
        labels_test (tensor): test labels
        mean_df (pd.DataFrame): dataframe containing X mean
        num_hours (int): num hours from mimic, should be 24
        seed (int): random seed
        verbose (bool, optional): flag for printoffs. Defaults to False.

    Returns:
        dict: confidence, audit score and attack score per target model per data point
    """
    global use_gpu, device_ids
    print(f'---Starting:\t{[ (k, v) for (k,v) in h.items() ]}')
    batch_size = int(h['batch_size'])

    np.random.seed(seed)
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


    shadow_all_inds = torch.randperm(len(all_label_train))
    # during training of grud we can only use a multiple of the batchsize
    # so we will use up to half the data points and the rest will be out
    N_in = ((len(shadow_all_inds)//2)//batch_size)*batch_size
    shadow_train_inds = shadow_all_inds[0:N_in]
    not_in_shadow_inds = shadow_all_inds[N_in:len(shadow_all_inds)]
    shadow_X_train = all_X_train[shadow_train_inds]
    shadow_label_train = all_label_train[shadow_train_inds]
    shadow_mean_df_ids = mean_df.index.get_level_values('subject_id').unique()[shadow_all_inds]
    shadow_x = mean_df.loc[shadow_mean_df_ids].mean()
    shadow_X_mean = torch.Tensor( np.array([shadow_x.values] *num_hours )).unsqueeze(0)
    if use_gpu:
        shadow_X_mean.to(device_type)

    shadow_base_params = {'X_mean': shadow_X_mean, 'input_size': shadow_X_mean.shape[2], 'device_id': device_type}
    shadow_base_params.update({k: v for k, v in h.items() 
                        if k in ('cell_size', 'hidden_size', 'batch_size', 'dropout_p')})
    if h["use_bn"] == "nobn":
        shadow_base_params.update({"use_bn": False})

    shadow_model = GRUD(**shadow_base_params)
    if use_gpu: shadow_model.to(device_type)

    if h["loss"] == "l2":
        def loss_fn(preds, targets):
            mse = torch.nn.MSELoss()
            return mse(torch.squeeze(preds), targets)
        label_type = torch.float32
        shadow_base_params.update({"apply_sigmoid" : True})
    elif h["loss"] == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        label_type = torch.float32
   
    # Make Pytorch Dataloader (batches) 
    shadow_train_dataloader = create_dataloader(shadow_X_train, shadow_label_train, batch_size=batch_size, label_type = label_type, drop_last = False)
    not_in_shadow_train_dataloader = create_dataloader(all_X_train[not_in_shadow_inds], all_label_train[not_in_shadow_inds], batch_size=batch_size, label_type = label_type, drop_last = False)
    test_dataloader = create_dataloader(X_test, label_test, batch_size=batch_size, label_type = label_type)

    best_shadow_model, _ = Train_Model(shadow_model, loss_fn, shadow_train_dataloader, 
        **{k: v for k, v in h.items() if k in (
            'num_epochs', 'patience', 'learning_rate', 'batch_size'
        )})
    # done training!
    
    # record metrics by adding them to the  hyperparms dict
    probabilities_dev, labels_dev = predict_proba(best_shadow_model, test_dataloader)
    y_score = np.concatenate(probabilities_dev).squeeze()
    targets  = np.concatenate(labels_dev).squeeze()

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
    
    print(f"---Finished: AUC = {h['auc']}")

    # compute confidences for those included in training and those that weren't
    if conf_type == "stable":
        in_confs = logit_confs_stable(best_shadow_model, shadow_train_dataloader).detach().cpu().numpy()
        out_confs = logit_confs_stable(best_shadow_model, not_in_shadow_train_dataloader).detach().cpu().numpy()
    elif conf_type == "hinge":
        in_confs = logit_confs_grud_hinge(best_shadow_model, shadow_train_dataloader).detach().cpu().numpy()
        out_confs = logit_confs_grud_hinge(best_shadow_model, not_in_shadow_train_dataloader).detach().cpu().numpy()


    mia_data_dict = {}
    # add to data dict
    for i in range(0, len(shadow_train_inds)):
        real_ind = int(shadow_train_inds[i])
        mia_data_dict[real_ind] = {"confs_in" : [], "confs_out" : []}
        mia_data_dict[real_ind]["confs_in"].append(in_confs[i])
    
    for i in range(0, len(not_in_shadow_inds)):
        real_ind = int(not_in_shadow_inds[i])
        mia_data_dict[real_ind] = {"confs_in" : [], "confs_out" : []}
        mia_data_dict[real_ind]["confs_out"].append(out_confs[i])
    
    return mia_data_dict
 


if __name__ == '__main__': 
#    losses = ['l2', 'bce']
    loss = 'bce'
    bns = ['nobn']#, 'nobn']
    targets = ['mort_icu', 'los_3']
    __spec__ = None
    ##### HYPERPARAMETER SEARCH SET UP
    # if use_full_train = True, the dev set will be added to train, and we will 
    # evaluate on the official test set (don't do this until the very end)
    use_full_train =  True
    ## hyperparameter sets
    h_pass =  'shadow_models'
    hyperparameter_set = non_priv_hyperparameter_set
    ####

    # set up multiprocessing
    if use_gpu:
        processes = n_procs * len(device_ids) # each process will be on one gpu
    else:
        processes = n_procs 
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)

    SEED = 1
    np.random.seed(SEED)

    FOLDER = Path(GRUD_MIA_RESULTS_FOLDER)
    run = f"{conf_type}_{N_shadow_models}"#"attempt_0405"
    RESULTS_FOLDER = Path(FOLDER, f"run_{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, will overwrite with new results...")
    RESULTS_FOLDER.mkdir(exist_ok=True)     

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
        mean_df =  (df_train.loc[:, pd.IndexSlice[:, 'mean']] * np.where((df_train.loc[:, pd.IndexSlice[:, 'mask']] == 1).values, 1, np.NaN))

        if use_gpu:
            X_train = dict(((gpu_id, X_train.to(gpu_id)) for gpu_id in device_ids))
            X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))
#            X_mean = dict(((gpu_id, X_mean.to(gpu_id)) for gpu_id in device_ids))
            label_train = dict(((gpu_id, label_train.to(gpu_id)) for gpu_id in device_ids))
            label_test = dict(((gpu_id, label_test.to(gpu_id)) for gpu_id in device_ids))

        for bn in bns:
            ## get the hyperparameters used in the final run of the baseline grud model
            # assumes that the run for the refined hyperparameter search for non-private
            # base line is 0.
            print(target)
            h = get_h_grud_nonpriv(target, 'shadow_models', loss, bn, 0, hyperparameter_set, use_full_train, 
                                   use_gpu, GRUD_BASELINE_RESULTS_FOLDER)[0]
                    

            args_list = [(h, X_train, label_train, X_test, label_test, mean_df, num_hours, seed, RESULTS_FOLDER) 
                        for seed in range(0, N_shadow_models)]

            # run the multiprocesing for l2 loss: 
            start = timer()
            print(f'------\n\nstarting training for {target} parallelized with {processes} cores ...\n\n----')
            
            records = pool.starmap(kernel_fn, args_list)# should return a list of the outputs 

            end = timer()
            
            print(f"\n\n---Runs for target = {target} done! \n\t {(end-start)/60} minutes to finish")
            print(f'Results saved in {RESULTS_FOLDER}')

            # save losses so we can see if training was wonky:
            df = pd.DataFrame.from_records(records)

            def combine_confs(lst):
                out = {"confs_in": [], "confs_out": []}
                for d in lst:
                    out["confs_in"] += d["confs_in"]
                    out["confs_out"] += d["confs_out"]
                out["confs_in_mean"] = np.array(out["confs_in"]).mean()
                out["confs_in_std"] = np.array(out["confs_in"]).std()
                out["confs_out_mean"] = np.array(out["confs_out"]).mean()
                out["confs_out_std"] = np.array(out["confs_out"]).std()
                return out
            
            final_mia_data_dict = df.apply(combine_confs).to_dict()
            results_folder = Path(RESULTS_FOLDER, "shadow_models", f"{h['target']}_loss_{h['loss']}_{bn}_results")
            results_folder.mkdir(exist_ok=True, parents=True)
            filename = f"mia_shadow_model_hps.json"
            path = Path(results_folder, filename)
            mia_filename = f"mia_shadow_confs.json"
            mia_path = Path(results_folder, mia_filename)
            print(f'\n\n---Kernel from run_mia_shadow_models.py: --------- Saving results data to {path}...')
            jsonify(h, path)
            jsonify(final_mia_data_dict, mia_path)
 
            print(f'\n---- Results for {target} ------')
            #print(final_mia_data_dict)
            print('\n-------------\n')
