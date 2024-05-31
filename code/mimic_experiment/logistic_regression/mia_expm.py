import torch, sys, pandas as pd, numpy as np, scipy
from pathlib import Path 
from torch.multiprocessing import Pool, set_start_method
sys.path.append(Path(".", "code").as_posix())
sys.path.append(Path(".", "code", "mimic_experiment").as_posix())
from config import *
from experiment_utils import *
from hyperparam_sets import expm_hyperparameter_set as hyperparameter_set, get_h_LR_expm

#### GPU SET UP
n_procs = 20 # if using gpus this will be per gpu
devices = {"proper_gpus": [0,2,3], "migs":[] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
    
# membership inference parameters
conf_type = "stable" # can be "stable" or "hinge"
total_target_models = 50

def kernel_fn(h, mia_data_dict, all_gpu_X_1, all_gpu_y_train, all_gpu_X_test, all_gpu_y_test, run_folder, use_gpu = use_gpu, device_ids = device_ids):
    """
    Args:
        h (dict): hyperparameter dict
        mia_data_dict: shadow confidences
        all_gpu_X_1 (tensor): the training data with a row of ones at the end (may or may not be on gpu)
        all_gpu_y_train (tensor): the training labels (may or may not be on gpu)
        all_gpu_X_test (tensor): the test data (may or may not be on gpu)
        all_gpu_y_test (tensor): the test labels (may or may not be on gpu)
        run_folder (string or pathlib.Path): where to store hyperparameter dict w/ AUC results added 
        use_gpu (bool): made above by setup_devices
        device_ids (list): made above by setup_devices
    Returns:
        dict: confidence, audit score and attack score per target model per data point
    """
    global conf_type
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
        all_X_1 = all_gpu_X_1[device_id]
        all_y_train = all_gpu_y_train[device_id]   
        all_X_train = all_X_1[0:len(all_y_train), 0:2496]
        X_test = all_gpu_X_test[device_id]
        y_test = all_gpu_y_test
    else:
        all_X_1 = all_gpu_X_1
        all_y_train = all_gpu_y_train   
        X_test = all_gpu_X_test
        y_test = all_gpu_y_test

    # used for auditing
    def score_loss_fn(y_pred, y):
        return (y_pred - y).pow(2)

    # we will only train on half the data
    ######### Train target model
    all_inds = torch.randperm(len(all_y_train))
    target_train_inds = all_inds[0:len(all_inds)//2]
    X_1 = all_X_1[target_train_inds]
    y_train = all_y_train[target_train_inds]

    yl = torch.stack([ y_train ]*batch_size, 1) # make labels for every y_pred value
    
    ## Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## setup Sylvester NF
    z_size = X_1.shape[1]
    model = NormalizingFlowSylvester(z_size, m = m, hh = hh, n_flows=n_flows) # number of layers/flows = n_flows
    # push model to gpu
    if use_gpu: model.to(device_id)

    # since we theorize that this approximates the exponential mechanism,
    # we will only sample and use one model from the NF to target/audit
    samples = random_normal_samples(1, z_size, sample_std, device = device_id) # l by X.shape[0] + 1 tensor, {u_i: i = 1,..,l} sampled from base. 
    initial_transformed_samples, _ = model.forward(samples) # l by X.shape[0] + 1 tensor, run thru NF 
    initial_beta = initial_transformed_samples.detach().squeeze()
    # make a logistic model from the sampled parameters and evaluate it
    target_model = get_logistic_model(initial_beta, X_test.shape[1], use_gpu = use_gpu, device_id = device_id)
    # privacy auditing
    score_before = score(target_model, score_loss_fn, all_X_train, all_y_train).detach().cpu().numpy()

    nf_train(model, X_1, yl, epsilon, s, lam, z_size, sample_std, learning_rate, 
                momentum, epochs, batch_size, patience, device_id)

    # eval: 
    transformed_samples, _ = model.forward(samples) # l by X.shape[0] + 1 tensor, run thru NF 
    beta = transformed_samples.detach().squeeze()
    # make a logistic model from the sampled parameters and evaluate it
    best_target_model = get_logistic_model(beta, X_test.shape[1], use_gpu = use_gpu, device_id = device_id)

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

    # privacy audit scores
    score_after = score(target_model, score_loss_fn, all_X_train, all_y_train).detach().cpu().numpy()
    scores = score_before - score_after

    ######## Attack Model 
    # Carlini et al. - 2022 - Membership Inference Attacks From First Principles
    
    # this dictionary will hold membership inference attack and privacy audit information 
    # for each data point in our training set (half were used to train the 
    # target model)
    target_data_dict = {} # hold mia results

    ### Membership inference attack
    if conf_type == "stable":
        confs_obs = logit_confs_stable(best_target_model, all_X_train, all_y_train).detach().cpu().numpy()
    elif conf_type == "hinge":
        confs_obs = logit_confs_lr_hinge(best_target_model, all_X_train, all_y_train).detach().cpu().numpy()
    for ind in range(0, len(all_X_train)):
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

    # write results: 
    results_folder = Path(run_folder, f"expm_{total_target_models}", "hyperparams", f'{target}_epsilon_{epsilon}_results')
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = f"model_{h['model_id']}.json"        
    path = Path(results_folder, filename)
    print(f'\n\n---Kernel from run_expm.py: ------Saving results data to {path}...')
    jsonify(h, path)
    ## print results    
    print(f"---Finished: AUC is {h['auc']}")

    return h, target_data_dict


if __name__ == '__main__': 
    # set up multiprocessing    
    ## for multiprocessing and using gpus
    if use_gpu:
        processes = n_procs * len(device_ids) # each process will be on one gpu
    else:
        processes = n_procs
    
    __spec__ = None #ipython is strange

    set_start_method('spawn', force = True)    
    pool = Pool(processes=processes)       
    ## seed
    SEED = 0
    np.random.seed(SEED)
    ## data set up: 
    data_norm = 'z'
    epsilons = [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] #1e-6, 1e-5, 
    h_pass = 'final' # we will use the best hyperparameters from the refined run
    use_full_train = True  
    N_shadow_models = 1000 # this will be used to load shadow model results
          
    ## find next folder number and make a new folder:     
    FOLDER = Path(LR_MIA_RESULTS_FOLDER)
    run = f"{conf_type}_{N_shadow_models}"
    RESULTS_FOLDER = Path(FOLDER, f"run_{run}")
    if RESULTS_FOLDER.exists(): 
        print(f"RESULTS_FOLDER {RESULTS_FOLDER} already exists, will write new results to existing folder...")
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


        for eps in epsilons:
            print("------\n\n Loading membership inference shadow model results ... \n\n------")
            mia_path = Path(RESULTS_FOLDER, "shadow_models", f"{target}_loss_bce_results", "mia_shadow_confs.json")
            mia_data_dict = pd.DataFrame.from_records(unjsonify(mia_path)).to_dict()

            print(f'\n---Script run_expm.py: Starting run = {run}, target = {target}---')
            h_list = get_h_LR_expm(target, h_pass, [eps], 0, 
                                   folder = LR_EXPM_RESULTS_FOLDER,
                                   hyperparameter_set=hyperparameter_set, 
                                   use_full_train=use_full_train, use_gpu = use_gpu, 
                                   data_norm = data_norm, final_n = total_target_models)
            args_list = [(h, mia_data_dict, X_1, y_train, X_test, y_test, RESULTS_FOLDER) 
                        for h in h_list]# list of arg tuples for kernel: 

            # run the multiprocesing 
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

            # write privacy audit and MIA results
            final_mia_data_dict = mia_df.apply(combine_confs).to_dict()
            results_folder = Path(RESULTS_FOLDER, f"expm_{total_target_models}", f"{target}_loss_l2_results")
            results_folder.mkdir(exist_ok=True, parents=True)
            mia_filename = f"mia_expm_epsilon_{eps}.json"
            mia_path = Path(results_folder, mia_filename)
            print(f'\n\n---Kernel from mia_expm.py: --------- Saving results data to {mia_path}...')
            jsonify(final_mia_data_dict, mia_path)


            df = pd.DataFrame.from_records(records)
            print(f"\n\n---Runs for {target} done!")
            print(f'Results saved in {RESULTS_FOLDER}')
            print(f"\n\n---- Printing results for {target}---\n")
            print(df)
            print('\n-------------\n')
