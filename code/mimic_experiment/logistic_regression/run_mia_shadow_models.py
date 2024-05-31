import pandas as pd
import numpy as np
import sys, copy
from pathlib import Path
from torch.multiprocessing import Pool, set_start_method#, cpu_count, current_process
sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from config import *
from experiment_utils import *
from hyperparam_sets import get_h_LR_nonpriv, non_priv_hyperparameter_set as hyperparameter_set

# GPU setup 
n_procs = 20 # if using gpus, this will be per gpu
devices = {"proper_gpus": [0, 1, 2], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
conf_type = "hinge" # "stable" or "hinge"
N_shadow_models = 1000


def kernel_fn(h, all_gpu_train_data, all_gpu_X_test, seed):
    """
        Trains shadow models on a random half of the data given and computes confidence scores for data
        included and not included in shadow model training

    Args:
        hyperparams (dict): dict of hyperperams, see unpacking of these params below for contents
        train_data (dict): dictionary of form {gpu_id/"cpu": training dataset maker (from load_data function)}
        test_data (dict): dictionary of form {gpu_id/"cpu": training dataset maker (from load_data function)}
        seed (int): random seed
    Returns
        dict: conf_in, conf_out per shadow model per data point and the respective means, std for each
    """
    global use_gpu, device_ids
    print(f'---Starting:\t{[ (k, v) for (k,v) in h.items() ]}')
    batch_size = int(h['batch_size'])
    np.random.seed(seed)
    if use_gpu:
        device_id = get_gpu_id(device_ids)
        all_train_data = all_gpu_train_data[device_id]
        X_test = all_gpu_X_test[device_id]
    else:
        all_train_data = all_gpu_train_data
        X_test = all_gpu_X_test
        device_id = 'cpu'

    # we will train on half of the data
    batch_size = int(h['batch_size'])
    shadow_all_inds = torch.randperm(len(all_train_data.y))
    shadow_train_inds = shadow_all_inds[0:len(shadow_all_inds)//2]
    not_in_shadow_inds = shadow_all_inds[len(shadow_all_inds)//2:len(shadow_all_inds)]
    shadow_train_data = DatasetMaker(all_train_data.X[shadow_train_inds], all_train_data.y[shadow_train_inds])
    shadow_loss_fn = loss_bce
    shadow_reduction = h['loss_reduction']

    shadow_model = LogisticRegression(X_test.shape[1], 1)
    if use_gpu: shadow_model.to(device_id)
    shadow_train_dataloader = DataLoader(shadow_train_data, batch_size, shuffle = False)

    best_shadow_model, _ = nonpriv_train(shadow_model, shadow_loss_fn, shadow_train_dataloader, 
        **{k: v for k, v in h.items() if k in (
            'num_epochs', 'patience', 'learning_rate', 'lam', 'momentum', 'lam', 
        )}, verbose = False, reduction = shadow_reduction)
    # done training!
    
    # compute confidences for those included in training and those that weren't
    y_in_score = best_shadow_model(shadow_train_data.X).squeeze().detach().cpu()
    y_out_score = best_shadow_model(all_train_data.X[not_in_shadow_inds]).squeeze().detach().cpu()
    
    if conf_type == "stable":
        in_confs = logit_confs_stable(best_shadow_model, shadow_train_data.X, shadow_train_data.y).detach().cpu().numpy()
        out_confs = logit_confs_stable(best_shadow_model, all_train_data.X[not_in_shadow_inds], all_train_data.y[not_in_shadow_inds]).detach().cpu().numpy()
    elif conf_type == "hinge":
        in_confs = logit_confs_lr_hinge(best_shadow_model, shadow_train_data.X, shadow_train_data.y).detach().cpu().numpy()
        out_confs = logit_confs_lr_hinge(best_shadow_model, all_train_data.X[not_in_shadow_inds], all_train_data.y[not_in_shadow_inds]).detach().cpu().numpy()

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
    ## find next folder number and make a new folder:     
    h_pass = 'shadow_models'
    use_full_train = True

    FOLDER = Path(LR_MIA_RESULTS_FOLDER)
    run = f"{conf_type}_{N_shadow_models}"#"attempt_0405"
    RESULTS_FOLDER = Path(FOLDER, f"run_{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, will overwrite with new results...")
    RESULTS_FOLDER.mkdir(exist_ok=True)     

    for target in targets: 
        _, _, _, _, train_data, test_data = load_data(target, level = 3, normalization = 'z',
                                                            use_full = use_full_train)
        X_test = test_data.X
        label_test = test_data.y

        if use_gpu: 
            train_data = dict(((gpu_id, train_data.to(gpu_id)) for gpu_id in device_ids))
            X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))

        for loss in ["bce"]:
            ## get the hyperparameters used in the final run of the baseline logistic model
            # assumes that the run for the refined hyperparameter search for non-private
            # base line is 0.
            h = get_h_LR_nonpriv(target, h_pass, loss, 0, hyperparameter_set, 
                                      use_full_train, use_gpu, LR_BASELINE_RESULTS_FOLDER)[0]
            
            
            print(f'------\nPreparing hyperparameters for training ...\n\n----')
           
            args_list = [(h, train_data, X_test, seed)
                            for seed in range(0, N_shadow_models)]
            
            print(f'------\n\nstarting training for {target, loss} parallelized with {processes} cores ...\n\n----')
            records = pool.starmap(kernel_fn, args_list)# should return a list of the outputs 
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
            # save shadow model results
            final_mia_data_dict = df.apply(combine_confs).to_dict()
            results_folder = Path(RESULTS_FOLDER, "shadow_models", f"{h['target']}_loss_{h['loss']}_results")
            results_folder.mkdir(exist_ok=True, parents=True)
            filename = f"mia_shadow_model_hps.json"
            path = Path(results_folder, filename)
            mia_filename = f"mia_shadow_confs.json"
            mia_path = Path(results_folder, mia_filename)
            print(f'\n\n---Kernel from run_mia_shadow_models.py: --------- Saving results data to {path}...')
            jsonify(h, path)
            jsonify(final_mia_data_dict, mia_path)
        
            print(f"\n\n---Runs for {target, loss} done!")
            print(f'Results saved in {RESULTS_FOLDER}')
            print(f"\n\n---- Printing results for {target, loss}---\n")
            print(df)
            print('\n-------------\n')




