from scipy.stats import chi2
import torch, sys, pandas as pd, numpy as np
from pathlib import Path 
from torch.multiprocessing import Pool, set_start_method
sys.path.append(Path(".", "code").as_posix())
sys.path.append(Path(".", "code", "mimic_experiment").as_posix())
sys.path.append(Path(".", "code", "mimic_experiment", "logistic_regression").as_posix())
from config import *
from experiment_utils import *
from hyperparam_sets import expm_hyperparameter_set as hyperparameter_set, get_h_LR_expm

#### GPU SET UP
devices = {"proper_gpus": [3], "migs":[] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
##### HYPERPARAMETER SEARCH SET UP ## hyperparameter pass used for folder set up and hyperparmeters to use: 
h_pass = 'final' # options are 'first', then 'refined' (both with use_full_train = False),  then the test run:  h_pass = 'final' and use_full_train = True
use_full_train = True # if use_full_train = True, the dev set will be added to train, and we will 
# evaluate on the official test set (don't do this until the end, i.e., h_pass = 'final')


## functions
def kernel_fn(h, X_1, y_train, X_test, X_test_1, y_test, run_folder, use_gpu = use_gpu, device_ids = device_ids):
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
    global results ## dict for results
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
        X_test_1 = X_test_1[device_id]
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

    ## eval
    samples = random_normal_samples(10000, z_size, sample_std, device = device_id) # l by X.shape[0] + 1 tensor, {u_i: i = 1,..,l} sampled from base. 
    yl = torch.stack([ y_train ]*10000, 1) # make labels for every y_pred value
    with torch.no_grad():
        B, _ = model.forward(samples) # l by X.shape[0] + 1 tensor, run thru NF 
    B.detach()
    
    # savepath =  Path(run_folder, "theta_samples.json")
    # jsonify(B.tolist(),savepath.as_posix())
    # print(f"10K Samples from trained LR model saved in {savepath}")
    
    print(f"B.shape = {B.shape}, X_test.shape[1] = {X_test.shape[1]}")
    
    results[epsilon] = { 
        'auc_baysian' : get_auc_bayesian(B, X_test_1, y_test), ## bayesian auc
        'auc_frequentist_1' : get_auc(B[0,:], X_test, y_test), # just use the first sample
        'auc_frequentist_map': get_auc(get_mode_MAP(B, X_1, yl, epsilon, lam),  X_test, y_test) ## AUC from the MAP
        }
    print(f"Results are {epsilon}: {results[epsilon]}")
    jsonify(results, Path(FOLDER, 'results.json'))
    print(f"Results saved in {Path(FOLDER, 'results.json')}")


def compute_credibility_region(B, lambda_reg = 1e-5):
    #B = torch.tensor(B, dtype=torch.float32)# Convert list of lists B to a PyTorch tensor
    # Compute the sample mean and covariance matrix
    mean = torch.mean(B, dim=0)
    cov = torch.cov(B.T, correction=1)
    
    # covariance with regularization to ensure invertibility
    regularized_cov = cov + lambda_reg * torch.eye(cov.shape[0])
    
    # Use pseudo-inverse if the dimensionality is high or the matrix is near-singular
    inv_cov = torch.linalg.pinv(regularized_cov)
        
    # Compute Mahalanobis distance for each sample
    centered_data = B - mean
    mahalanobis_distance = torch.diag(torch.mm( torch.mm(centered_data, inv_cov), centered_data.T))
    # m_distance^2 follows a chi^2. 
    # Determine the 95% threshold using the chi-squared distribution
    threshold = chi2.ppf(0.95, df=B.shape[1])
    # Determine which samples are within the 95% credibility region
    within_region = (mahalanobis_distance <= threshold)
    # The region is implicitly defined by these parameters
    return {'mean': mean, 
            'covariance': cov, 
            'threshold': threshold, 
            'within_region': within_region }
    

def get_auc_bayesian(B, X_test_1, y_test):
    ## inference:
    ## given x, predict $E_y(y|x) = E_\theta( p(y=1|x, \theta) )~= (1/N) \sum_i p(y=1|x, \theta)
    ## p(y=1|x, \theta) =  \exp(-\epsilon l(x,y=1,t_i)/2)/Z$
    ## Z = \exp(-\epsilon l(x,y=1,t_i)/2) + \exp(-\epsilon l(x,y=0,t_i)/2)
    ## l(x,y = 1, t_i) = (1-sigmoid(x.T* t_i))^2
    # B.to('cpu')
    ## X is k by m, B is n by m. Want matrix X*B^T  = [ x[i,:]^t b[j,:] ]_i,j, k by n 
    XBt = torch.mm(X_test_1, B.T)
    Z_1 = torch.exp(-(epsilon/2.) * (1-torch.sigmoid(XBt))**2)
    Z_0 = torch.exp(-(epsilon/2.) * (0-torch.sigmoid(XBt))**2)
    y_preds = (Z_1/(Z_1 + Z_0)).mean(dim = 1)
    return roc_auc_score(y_test.to('cpu'), y_preds.to('cpu'))


def get_mode_MAP(B, X_1, yl, epsilon, lam, s = 1):
    """MAP estimate: 
    Given samples n samples from trained NF (B is n samples by l), 
    this returns the sample (B[i,:]) with highest likelihood (the mode or MAP)
    """
    y_preds = torch.sigmoid(torch.mm(X_1,B.T)) # shape is (n by l): \phi(x^t b)  for each of the n data points x\in X, and for each of t>
    ## compute MSE: 
    losses = (y_preds - yl).pow(2) # shape is n by l (loss of all n data points for each of l betas)
    losses = losses.sum(0) #sum down the rows, should have shape [l]
    ## regularization:
    regs = B.abs()
    regs = regs.sum(axis = 1) # shape is l
    ## sum and negate to get utility: 
    util_l2 = - (losses + (lam * regs))
    ## finally, we have the target potential function: 
    log_p_x = util_l2 * epsilon/(2*s)
    i = np.where(log_p_x.cpu() == max(log_p_x.cpu()))[0][0] ## index where max is attained
    return B[i]
    
## setup for training: 
target = "mort_icu"
data_norm = 'z'

_, _, _, _, train_data, test_data = load_data(target, level = 3, normalization = 'z',
                                                    use_full = use_full_train)
X_train =  train_data.X
X_1 = torch.concat([X_train, torch.ones([X_train.shape[0], 1]) ], axis = 1) # adds 1s for last column 
y_train = train_data.y.squeeze()
X_test = test_data.X
y_test = test_data.y.squeeze()
X_test_1 = torch.concat([X_test, torch.ones([X_test.shape[0], 1]) ], axis = 1) # adds 1s for last column 

if use_gpu: # put data on gpu, keep dict of what tensors are where (to be access by child process)
    X_1 = dict(((gpu_id, X_1.to(gpu_id)) for gpu_id in device_ids))
    y_train = dict(((gpu_id, y_train.to(gpu_id)) for gpu_id in device_ids))
    X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))
    y_test = dict(((gpu_id, y_test.to(gpu_id)) for gpu_id in device_ids))
    X_test_1 = dict(((gpu_id, X_test_1.to(gpu_id)) for gpu_id in device_ids))
        

FOLDER = Path(LR_EXPM_RESULTS_FOLDER,  f"mort_uq") 
FOLDER.mkdir(exist_ok=True)

results = { }
for epsilon in [.001, .01, .1, 1]:
    run_folder = Path(FOLDER,f"epsilon_{epsilon}")
    run_folder.mkdir(exist_ok=True) 

    print(f'\n---Starting to train an LR model on target = {target} for epsilon = {epsilon}---')
    ## get hyperparms: 
    h = get_h_LR_expm(target, h_pass, [epsilon], 0, 
                            hyperparameter_set=hyperparameter_set, 
                            use_full_train=use_full_train, use_gpu = use_gpu, 
                            data_norm = data_norm)[0]
    kernel_fn(h, X_1, y_train, X_test, X_test_1, y_test, run_folder, use_gpu = use_gpu, device_ids = device_ids)

## make latex table:     
df = pd.DataFrame(unjsonify(resultspath))
pd.options.display.float_format = "{:,.2f}".format ## only two decimal places
print(df.to_latex())