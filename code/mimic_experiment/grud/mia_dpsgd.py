import torch, logging, warnings, sys, scipy, pandas as pd, numpy as np, copy
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
n_procs = 20 # if using gpus, this will be per gpu
devices = {"proper_gpus": [0,2,3], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
conf_type = "stable" # can be "stable" or "hinge"
total_target_models = 50

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
 
 
# requires shadow models have been trained before running this
## runs mia and privacy for a set of hyperparameters on given data
def kernel_fn(h, mia_data_dict, all_gpu_X_train, all_gpu_label_train, all_gpu_X_test, 
              all_gpu_label_test, mean_df, num_hours, run_folder): 
    """runs training and test results (on dev_data) of dpsgd GRUD with given hyperparameters

    Args:
        hyperparams (dict): dict of hyperperams, see unpacking of these params below for contents
        X_train (tensor): train inputs
        label_train (tensor): train labels
        X_test (tensor): test inputs
        labels_test (tensor): test labels
        mean_df (pd.DataFrame): dataframe containing X mean
        num_hours (int): num hours from mimic, should be 24
        run_folder (str): where to save

    Returns:
        dict with results and hyperparams: 
        list of training losses from training 
        model : pythorch model trained
    """
    global use_gpu, device_ids
    print(f'---Starting:\t{[ (k, v) for (k,v) in h.items() ]}')
#    early_stop_frac, batch_size, seed = [h[k] for k in ('early_stop_frac','batch_size','seed')]
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

    computed_X_mean = torch.Tensor( np.array([target_x.values] *num_hours )).unsqueeze(0)
    target_X_mean = torch.zeros(computed_X_mean.shape)
    assert (abs(computed_X_mean) < 1e-10).all() , "Computed X_mean is not 0. Ensure data is normalized to have mean 0"
    # grud drops the extra data points, but we need them so we will make another batch at the end using the first
    # values and just not use them...
    diff = len(target_all_inds) - len(target_all_inds)//batch_size*batch_size 
    if diff != 0:
        to_add = batch_size - diff
        bigger_X_train = torch.concat((all_X_train, all_X_train[range(0,to_add)]))
        bigger_label_train = torch.concat((all_label_train, all_label_train[range(0,to_add)]))
        all_dataloader = create_dataloader(bigger_X_train, bigger_label_train, batch_size=batch_size)
    else:
        all_dataloader = create_dataloader(all_X_train, all_label_train, batch_size=batch_size)



    # Make Pytorch Dataloader (batches) 
    target_train_dataloader = create_dataloader(target_X_train, target_label_train, batch_size=batch_size)
    test_dataloader = create_dataloader(X_test, label_test, batch_size=batch_size)

    sample_rate = 1/len(target_train_dataloader) # already incorporates batchsize
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
    target_base_params = {'X_mean': target_X_mean, 'input_size': target_X_mean.shape[2], 'device_id': device_type}
    target_base_params.update({k: v for k, v in h.items() 
                        if k in ('cell_size', 'hidden_size', 'batch_size', 'dropout_p')})
    if h["use_bn"] == "nobn":
        target_base_params.update({"use_bn": False})

    if h["loss"] == "l2":
        def loss_fn(preds, targets):
            mse = torch.nn.MSELoss()
            return mse(torch.squeeze(preds), targets)
        target_base_params.update({"apply_sigmoid" : True})
        score_loss_fn = torch.nn.MSELoss(reduce = False) # privacy auditing
    elif h["loss"] == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        score_loss_fn = torch.nn.BCEWithLogitsLoss(reduce = False) # privacy auditing

    target_model = GRUD(**target_base_params)
    if use_gpu:
        target_model.to(device_type)

    # privacy auditing
    score_before = score(copy.deepcopy(target_model), score_loss_fn, all_dataloader)

    best_target_model, niter_per_epoch, privacy_engine = Train_Model_DPSGD(
        target_model, loss_fn, target_train_dataloader, noise_multiplier, 
        **{k: v for k, v in h.items() if k in (
            'num_epochs', 'patience', 'learning_rate', 'batch_size', 'max_grad_norm'
        )})
    # done training!
    
    # record metrics by adding them to the  hyperparms dict
    probabilities_dev, labels_dev = predict_proba(best_target_model, test_dataloader)
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
    
    ######## Target model training complete

    # privacy auditing
    score_after = score(best_target_model, score_loss_fn, all_dataloader)
    scores = (score_before - score_after).cpu().numpy()
    ######## Train shadow models following 
    # Carlini et al. - 2022 - Membership Inference Attacks From First Principles
    
    # this dictionary will hold membership inference attack information 
    target_data_dict = {}
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


    print(f'\n\n---Script run_dpsgd_grud_los_mort.py: --------- Saving results data ...')
    results_folder = Path(run_folder, f"dpsgd_{total_target_models}", "hyperparams", f"{h['target']}_epsilon_{h['target_epsilon']}_loss_{h['loss']}_{h['use_bn']}_results")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = f"model_{h['model_id']}.json"
    jsonify(h, Path(results_folder, filename))
    print(f"---Finished model {h['model_id']}: AUC = {h['auc']} with epsilon = {h['prv_epsilon']}")
    
    return h, target_data_dict


if __name__ == '__main__': 
    epsilons = [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] # epsilons to run over
    delta = 1e-5
    targets = ['mort_icu', 'los_3']
    bn = "nobn"
    SEED = 1
    if use_gpu:
        processes = n_procs * len(device_ids) # each process will be on one gpu
    else:
        processes = n_procs
 
    # set up multiprocessing
    set_start_method('spawn', force = True)
    pool = Pool(processes=processes)
    
    use_full_train = True
    h_pass = 'final' 
    hyperparameter_set = dpsgd_hyperparameter_set
    N_shadow_models = 1000

    FOLDER = Path(GRUD_MIA_RESULTS_FOLDER)
    FOLDER.mkdir(exist_ok=True) 
    ## find next folder number and make a new folder:     
    run = f"{conf_type}_{N_shadow_models}"
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
            label_train = dict(((gpu_id, label_train.to(gpu_id)) for gpu_id in device_ids))
            label_test = dict(((gpu_id, label_test.to(gpu_id)) for gpu_id in device_ids))
            
        np.random.seed(SEED)
        print("------\n\n Loading membership inference shadow model results ... \n\n------")
        mia_path = Path(RESULTS_FOLDER, "shadow_models", f"{target}_loss_bce_{bn}_results", "mia_shadow_confs.json")
        mia_data_dict = pd.DataFrame.from_records(unjsonify(mia_path)).to_dict()


        for loss in ['l2', 'bce']:
            for epsilon in epsilons:
                h_list = get_h_grud_dpsgd(target, h_pass, loss, bn, epsilon, delta,
                                                   "0", hyperparameter_set, use_full_train, use_gpu, 
                                                    GRUD_DPSGD_RESULTS_FOLDER, final_n = total_target_models)
                
                args_list = [(h, mia_data_dict, X_train, label_train, X_test, label_test, 
                              mean_df, num_hours, RESULTS_FOLDER) for h in h_list]
         
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
                results_folder = Path(RESULTS_FOLDER, f"dpsgd_{total_target_models}", f"{target}_loss_{loss}_{bn}_results")
                results_folder.mkdir(exist_ok=True, parents=True)
                mia_filename = f"mia_dpsgd_epsilon_{epsilon}.json"
                mia_path = Path(results_folder, mia_filename)
                
                print(f'Results saved in {mia_path}')
                jsonify(final_mia_data_dict, mia_path)

                print(f"\n\n---- Printing results for {target}---\n")
                df = pd.DataFrame.from_records(records)
                print(f'\n---- Results for {target} ------')
                print(df)
                print('\n-------------\n')
