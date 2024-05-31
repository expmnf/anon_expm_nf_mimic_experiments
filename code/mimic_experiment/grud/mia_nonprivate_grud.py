import pandas as pd
import numpy as np
import torch, logging, warnings, sys, itertools, scipy
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
n_procs = 20 # if using gpus, this will be per gpu
devices = {"proper_gpus": [0,2,3], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)

# Membership Inference hyperparameters
conf_type = "stable" # can be "stable" or "hinge"
total_target_models = 50

# requires shadow models have been trained before running this
## runs mia and privacy for a set of hyperparameters on given data
def kernel_fn(h, mia_data_dict, all_gpu_X_train, all_gpu_label_train, all_gpu_X_test, 
              all_gpu_label_test, mean_df, num_hours, run_folder): 
    """runs training and test results (on dev_data) of GRUD with given hyperparameters

    Args:
        hyperparams (dict): dict of hyperperams, see unpacking of these params below for contents
        mia_data_dict (dict): shadow confidences
        X_train (tensor): train inputs
        label_train (tensor): train labels
        X_test (tensor): test inputs
        labels_test (tensor): test labels
        mean_df (pd.DataFrame): dataframe containing X mean
        num_hours (int): num hours from mimic, should be 24
        run_folder (str): where to save

    Returns:
        dict with results and hyperparams: 
        dict: confidence, audit score and attack score per target model per data point
    """
    global use_gpu, device_ids, conf_type
    print(f'---Starting:\t{[ (k, v) for (k,v) in h.items() ]}')
    batch_size, seed = [h[k] for k in ('batch_size','seed')]

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

    target_all_inds = torch.randperm(len(all_label_train))
    # during training of grud we can only use a multiple of the batchsize
    # so we will use up to half the data points and the rest will be out
    N_in = ((len(target_all_inds)//2)//batch_size)*batch_size
    target_train_inds = target_all_inds[0:N_in]
    target_X_train = all_X_train[target_train_inds]
    target_label_train = all_label_train[target_train_inds]
    target_mean_df_ids = mean_df.index.get_level_values('subject_id').unique()[target_all_inds]
    target_x = mean_df.loc[target_mean_df_ids].mean()
    target_X_mean = torch.Tensor( np.array([target_x.values] *num_hours )).unsqueeze(0)
    if use_gpu:
        target_X_mean.to(device_type)

    
    target_base_params = {'X_mean': target_X_mean, 'input_size': target_X_mean.shape[2], 'device_id': device_type}
    target_base_params.update({k: v for k, v in h.items() 
                        if k in ('cell_size', 'hidden_size', 'batch_size', 'dropout_p')})
    if h["use_bn"] == "nobn":
        target_base_params.update({"use_bn": False})

    if h["loss"] == "l2":
        def loss_fn(preds, targets):
            mse = torch.nn.MSELoss()
            return mse(torch.squeeze(preds), targets)
        label_type = torch.float32
        target_base_params.update({"apply_sigmoid" : True})
        score_loss_fn = torch.nn.MSELoss(reduce = False) # privacy auditing
    elif h["loss"] == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        label_type = torch.float32
        score_loss_fn = torch.nn.BCEWithLogitsLoss(reduce = False) #privacy auditing

    target_model = GRUD(**target_base_params)
    if use_gpu: target_model.to(device_type)
   
    # grud drops the extra data points, but we need them so we will make another batch at the end using the first
    # values and just not use them...
    diff = len(target_all_inds) - len(target_all_inds)//batch_size*batch_size 
    if diff != 0:
        to_add = batch_size - diff
        bigger_X_train = torch.concat((all_X_train, all_X_train[range(0,to_add)]))
        bigger_label_train = torch.concat((all_label_train, all_label_train[range(0,to_add)]))
        all_dataloader = create_dataloader(bigger_X_train, bigger_label_train, batch_size=batch_size, label_type = label_type)
    else:
        all_dataloader = create_dataloader(all_X_train, all_label_train, batch_size=batch_size, label_type = label_type)

    score_before = score(target_model, score_loss_fn, all_dataloader)

    # Make Pytorch Dataloader (batches) 
    target_train_dataloader = create_dataloader(target_X_train, target_label_train, batch_size=batch_size, label_type = label_type)
    test_dataloader = create_dataloader(X_test, label_test, batch_size=batch_size, label_type = label_type)

    best_target_model, _ = Train_Model(target_model, loss_fn, target_train_dataloader, 
        **{k: v for k, v in h.items() if k in (
            'num_epochs', 'patience', 'learning_rate', 'batch_size'
        )})
    # done training!
    
    # record metrics by adding them to the  hyperparms dict
    probabilities_dev, labels_dev = predict_proba(best_target_model, test_dataloader)
    y_score = np.concatenate(probabilities_dev)
    targets  = np.concatenate(labels_dev)

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
    

    ######## Target model training complete

    ######## Train shadow models following 
    # Carlini et al. - 2022 - Membership Inference Attacks From First Principles
    
    # this dictionary will hold membership inference attack information 
    target_data_dict = {}
    score_after = score(best_target_model, score_loss_fn, all_dataloader) # privacy auditing
    scores = (score_before - score_after).cpu().numpy() # privacy auditing
    # for each data point in our training set (half were used to train the 
    # target model)
    # compute confidences for those included in training and those that weren't
    
    if conf_type == "stable":
        confs_obs = logit_confs_stable(best_target_model, all_dataloader).detach().cpu().numpy()
    elif conf_type == "hinge":
        confs_obs = logit_confs_grud_hinge(best_target_model, all_dataloader).detach().cpu().numpy()
    
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


    print(f'\n\n---Script run_nonprivate_grud_los_mort.py: --------- Saving results data ...')
    results_folder = Path(run_folder, f"nonprivate_{total_target_models}", "hyperparams", f"{h['target']}_loss_{h['loss']}_{h['use_bn']}_results")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = f"model_{h['model_id']}.json"
    jsonify(h, Path(results_folder, filename))
    print(f"---Finished: AUC = {h['auc']}")
    return h, target_data_dict#, losses[0], losses[1]#, best_model.cpu() 


if __name__ == '__main__': 
    losses = ['l2', 'bce']
    bns = ['bn', 'nobn']
    targets = ['mort_icu', 'los_3']

    # set up multiprocessing
    if use_gpu:
        processes = n_procs * len(device_ids) # each process will be on one gpu
    else:
        processes = n_procs 
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)

    SEED = 1
    np.random.seed(SEED)
    use_full_train =  False
    h_pass =  'final'
    N_shadow_models = 1000
    hyperparameter_set = non_priv_hyperparameter_set

    FOLDER = Path(GRUD_MIA_RESULTS_FOLDER)
    run = f"{conf_type}_{N_shadow_models}"
    RESULTS_FOLDER = Path(FOLDER, f"run_{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, will overwrite with new results...")
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
            label_train = dict(((gpu_id, label_train.to(gpu_id)) for gpu_id in device_ids))
            label_test = dict(((gpu_id, label_test.to(gpu_id)) for gpu_id in device_ids))

        for bn in bns:
            print("------\n\n Loading membership inference shadow model results ... \n\n------")
            mia_path = Path(RESULTS_FOLDER, "shadow_models", f"{target}_loss_bce_{bn}_results", "mia_shadow_confs.json")
            mia_data_dict = pd.DataFrame.from_records(unjsonify(mia_path)).to_dict()

            for loss in losses:
                # we will test on bce and l2 loss with and without batchnorm
                h_list = get_h_grud_nonpriv(target, h_pass, loss, bn, 0, 
                                            hyperparameter_set, use_full_train, use_gpu, 
                                            GRUD_BASELINE_RESULTS_FOLDER, final_n = total_target_models)

                args_list = [(h, mia_data_dict, X_train, label_train, X_test, label_test, mean_df, num_hours, RESULTS_FOLDER) 
                            for h in h_list]


                # run the multiprocesing for l2 loss: 
                start = timer()
                print(f'------\n\nstarting training for {target} parallelized with {processes} cores ...\n\n----')
                
                records = pool.starmap(kernel_fn, args_list)# should return a list of the outputs 
                end = timer()
                print(f"\n\n---Runs for target = {target} done! \n\t {(end-start)/60} minutes to finish")
                
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
                results_folder = Path(RESULTS_FOLDER, f"nonprivate_{total_target_models}", f"{target}_loss_{loss}_{bn}_results")
                results_folder.mkdir(exist_ok=True, parents=True)
                mia_filename = f"mia_nonprivate_baseline.json"
                mia_path = Path(results_folder, mia_filename)
                
                print(f'Results saved in {mia_path}')
                jsonify(final_mia_data_dict, mia_path)

                # save losses so we can see if training was wonky:
                print(f"\n\n---- Printing results for {target}---\n")
                df = pd.DataFrame.from_records(records)
                print(f'\n---- Results for {target} ------')
                print(df)
                print('\n-------------\n')
