import pandas as pd
import numpy as np
import sys, scipy
from pathlib import Path
from torch.multiprocessing import Pool, set_start_method#, cpu_count, current_process
sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from config import *
from experiment_utils import *
from hyperparam_sets import get_h_LR_nonpriv, non_priv_hyperparameter_set as hyperparameter_set

# GPU setup 
n_procs = 20# if using gpus, this will be per gpu
devices = {"proper_gpus": [0, 2, 3], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)

# Membership Inference hyperparameters
conf_type = "stable" # can be "stable" or "hinge"
total_target_models = 50

# requires shadow models have been trained before running this
## for accuracy results, this runs results for a set of hyperparameters on given data
def kernel_fn(h, mia_data_dict, all_gpu_train_data, all_gpu_X_test, y_test, run_folder, pool = None,
              total_target_models = total_target_models): 
    """
        trains a model
        runs training and test results (on dev_data) with given hyperparameters
    Args:
        hyperparams (dict): dict of hyperperams, see unpacking of these params below for contents
        mia_data_dict (dict): shadow confidences
        train_data (dict): dictionary of form {gpu_id/"cpu": training dataset maker (from load_data function)}
        X_test (tensor): test inputs
        labels_test (tensor): test labels
        run_folder (Path or string): folder for results
        use_gpu (bool): made above by setup_devices
        device_ids (list): made above by setup_devices
    Returns:
        h: updated hyperparm dict with results and hyperparams
        dict: confidence, audit score and attack score per target model per data point
    """
    global use_gpu, device_ids, conf_type
    print(f'---Starting:\t{[ (k, v) for (k,v) in h.items() ]}')
    batch_size = int(h['batch_size'])
    seed = h['seed']
    np.random.seed(seed)
    if use_gpu:
        device_id = get_gpu_id(device_ids)
        all_train_data = all_gpu_train_data[device_id]
        X_test = all_gpu_X_test[device_id]
    else:
        all_train_data = all_gpu_train_data
        X_test = all_gpu_X_test
        device_id = 'cpu'

    target_data_dict = {} # hold mia results

    # we will only train on half the data
    ######### Train target model
    all_inds = torch.randperm(len(all_train_data.y))
    target_train_inds = all_inds[0:len(all_inds)//2]
    target_train_data = DatasetMaker(all_train_data.X[target_train_inds], all_train_data.y[target_train_inds])
   
    if h["loss"] == "l2":
        target_loss_fn = loss_l2
        score_loss_fn = torch.nn.BCELoss(reduction = 'none' )     
    elif h["loss"] == "bce":
        target_loss_fn = loss_bce
        def score_loss_fn(y_pred, y):
            return (y_pred - y).pow(2)
    target_reduction = h['loss_reduction']
    
    target_model = LogisticRegression(X_test.shape[1], 1)
    if use_gpu: target_model.to(device_id)
    # Make Pytorch Dataloader (batches) 
    target_train_dataloader = DataLoader(target_train_data, batch_size, shuffle = False)

    # privacy auditing
    score_before = score(target_model, score_loss_fn, all_train_data.X, all_train_data.y).detach().cpu().numpy()

    best_target_model, _ = nonpriv_train(target_model, target_loss_fn, target_train_dataloader, 
        **{k: v for k, v in h.items() if k in (
            'num_epochs', 'patience', 'learning_rate', 'lam', 'momentum', 'lam', 
        )}, verbose = False, reduction = target_reduction)
    # done training!
    
    # record metrics by adding them to the  hyperparms dict
    y_score = best_target_model(X_test).squeeze().detach().cpu()

    h['auc'] = roc_auc_score(y_test, y_score)
    h['aps'] = average_precision_score(y_test, y_score)
    
    # get best threhold 
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresh":thresholds})
    df['dist_to_[0,1]'] = df.apply( lambda row: np.sqrt((1-row.tpr)**2 + (row.fpr)**2), axis = 1) # dist to [0,1]
    best = df[ df['dist_to_[0,1]'] == df['dist_to_[0,1]'].min()].iloc[0] # closest point on ROC to [0,1]
    thresh = best.thresh
    y_pred = (y_score>thresh) # binary predictions

    # metrics at "best" threshold
    best['accuracy'] = accuracy_score(y_test, y_pred)
    best['f1']  = f1_score(y_test, y_pred)
    best['prec'] = precision_score(y_test, y_pred)
    
    ######## Target model training complete
    # privacy auditing
    score_after = score(best_target_model, score_loss_fn, all_train_data.X, all_train_data.y).detach().cpu().numpy()
    scores = score_before - score_after

    ######## Train shadow models following 
    # Carlini et al. - 2022 - Membership Inference Attacks From First Principles
    
    # this dictionary will hold membership inference attack information 
    # for each data point in our training set (half were used to train the 
    # target model)

    if conf_type == "stable":
        confs_obs = logit_confs_stable(best_target_model, all_train_data.X, all_train_data.y).detach().cpu().numpy()
    elif conf_type == "hinge":
        confs_obs = logit_confs_lr_hinge(best_target_model, all_train_data.X, all_train_data.y).detach().cpu().numpy()
    for ind in range(0, len(all_train_data.X)):
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

    # add to results: 
    h.update(best.to_dict())
    
    results_folder = Path(run_folder, f"nonprivate_{total_target_models}", "hyperparams", f"{h['target']}_loss_{h['loss']}_results")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = f"model_{h['model_id']}.json"
    path = Path(results_folder, filename)
    print(f'\n\n---Kernel from run_nonprivate.py: --------- Saving results data to {path}...')
    jsonify(h, path)
    print(f"---Finished: AUC = {h['auc']}")
    return h, target_data_dict #, losses[0], losses[1]#, best_model.cpu() 


if __name__ == '__main__': 
    # set up multiprocessing
    ## for multiprocessing and using gpus
    if use_gpu:
        processes = n_procs * len(device_ids) # each process will be on one gpu
    else:
        processes = n_procs
    
    __spec__ = None
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)
    ## seed
    SEED = 1
    np.random.seed(SEED)
    h_pass = 'final' # we will use the best hyperparameters from refined run
    use_full_train = True
    N_shadow_models = 1000

    FOLDER = Path(LR_MIA_RESULTS_FOLDER)
    run = f"{conf_type}_{N_shadow_models}"
    RESULTS_FOLDER = Path(FOLDER, f"run_{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, overwriting previous results...")
    RESULTS_FOLDER.mkdir(exist_ok=True)     

    for target in targets: 
        _, _, _, _, train_data, test_data = load_data(target, level = 3, normalization = 'z',
                                                            use_full = use_full_train)
        X_test = test_data.X
        label_test = test_data.y
        
        if use_gpu: 
            train_data = dict(((gpu_id, train_data.to(gpu_id)) for gpu_id in device_ids))
            X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))

        for loss in losses:
            ## get the hyperparameters corresponding to this target and pass (defined above)                    
            h_list = get_h_LR_nonpriv(target, h_pass, loss, 0, hyperparameter_set, use_full_train, use_gpu, LR_BASELINE_RESULTS_FOLDER, final_n = total_target_models)
            
            print("------\n\n Loading membership inference shadow model results ... \n\n------")
            mia_path = Path(RESULTS_FOLDER, "shadow_models", f"{target}_loss_bce_results", "mia_shadow_confs.json")
            mia_data_dict = pd.DataFrame.from_records(unjsonify(mia_path)).to_dict()
            args_list = [(h, mia_data_dict, train_data, X_test, label_test, RESULTS_FOLDER) 
                            for h in h_list]
            
            print(f'------\n\nstarting training for {target, loss} parallelized with {processes} cores ...\n\n----')
            records = pool.starmap(kernel_fn, args_list)# should return a list of the outputs 
            mia_data_dicts = [r[1] for r in records]
            mia_df = pd.DataFrame.from_records(mia_data_dicts)

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
            # save membership inference and privacy auditing results
            final_mia_data_dict = mia_df.apply(combine_confs).to_dict()
            results_folder = Path(RESULTS_FOLDER, f"nonprivate_{total_target_models}", f"{target}_loss_{loss}_results")
            results_folder.mkdir(exist_ok=True, parents=True)
            mia_filename = f"mia_nonprivate_baseline.json"
            mia_path = Path(results_folder, mia_filename)
            print(f'\n\n---Kernel from mia_nonprivate.py: --------- Saving results data to {mia_path}...')
            jsonify(final_mia_data_dict, mia_path)
 
        
            print(f"\n\n---Runs for {target, loss} done!")
            print(f'Results saved in {RESULTS_FOLDER}')
            print(f"\n\n---- Printing results for {target, loss}---\n")
            df = pd.DataFrame.from_records(records)
            print(df)
            print('\n-------------\n')


#
#[-0.10772755833944192]
#[-0.14823946043777225]
#[0.2636509196632639]
#[-0.3286966874001078]
#[0.38715990257857813]
#[0.221381531501593]
#[-0.12295052839472342]
#[0.09563304828222519]
