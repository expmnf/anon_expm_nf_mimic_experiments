from pathlib import Path 
import scipy.stats as ss
import sys, itertools
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, accuracy_score, f1_score, precision_score
import numpy as np
from pprint import pformat
from torch.multiprocessing import current_process
from torch.optim.lr_scheduler import ReduceLROnPlateau
from opacus.accountants.utils import get_noise_multiplier
from opacus import PrivacyEngine
sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from data_utils import * 
from config import * 
from regression_models import LogisticRegression
from normalizing_flows import * 

idx = pd.IndexSlice


## GPU setup functions
def setup_devices(devices): #### GPU setup automated
    """prepares gpu_names and device_ids lists from available gpus and sets up gpus for pytorch
    args: 
        devices (dict): dict of form {"proper_gpus": [1, 2], "migs": ['mig2']}
    returns
        gpu_names (list of string), gpu_ids (list(range(number of gpus))), use_gpu (bool)
    """        
    mig_lookup = {  'mig0': "MIG-5c8ce46c-4654-56b3-9a17-262f306d871c",
                    'mig1': "MIG-b5127348-4e38-5e09-8ec9-df2ae46c3433",
                    'mig2': "MIG-55cbb0eb-f976-5544-95a9-35df621c5f87",
                    'mig3': "MIG-a332812e-a264-58ee-8e1e-dec4f6e4f36b"
                    }
    gpu_names = devices['proper_gpus'] + [mig_lookup[x] for x in devices['migs']] # actual names of all GPUs to use
    gpu_ids = list( range( len( gpu_names ) ) )   # indices starting at 0 of the GPUs we'll use
    num_gpus_available =  len(gpu_ids)
    if num_gpus_available > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([f'{name}' for name in gpu_names])
    
    use_gpu = False #this automatically sets a gpu flag: 
    if len(devices["proper_gpus"]) + len(devices["migs"]) > 0: 
        use_gpu = True 
    return gpu_names, gpu_ids, use_gpu


def get_gpu_id(device_ids):
    ngpus = len(device_ids) 
    cpu_name = current_process().name
    if cpu_name == "MainProcess": return 0
    cpu_id = int( cpu_name[cpu_name.find('-') + 1:]) - 1
    if ngpus > 1:
        gpu_id = device_ids[cpu_id%ngpus]
    else:
        gpu_id = device_ids[0]
    return gpu_id


## loading data functions: 
def load_data(target, level = 3, normalization = 'l_inf', device_type = 'cpu', use_full = False): 
    """reads from disk the mimic-iii processed data. 
    (See /notebooks/mimic_preprocessing for info on preprocessing)

    Args:
        target (string): 'mort_icu', or  'los_3'
        level (int, optional): 2 (unpivoted) or 3 (pivoted) Defaults to 3.
        normalization (str, optional): 'l_inf' or 'l_z'. Defaults to 'l_inf'.
        device_type (str, int): "cpu" or gpu_id
        use_full (bool): False gives train/dev split. True gives train+dev/test split. 
    Returns:
        returns 6-tuple, 
        - first two objects are dataframes (train, test), 
        - next two are targets (train, test), 
        - next two are the DataMaker class objects (train, test), 
    """
    # read in targets: 
    Ys_train = pd.read_csv( Path(DATA_FOLDER, 'Ys_train.csv'),  index_col=[0,1,2])
    Ys_test = pd.read_csv(Path(DATA_FOLDER, 'Ys_test.csv'),  index_col=[0,1,2])
    Ys_dev = pd.read_csv(Path(DATA_FOLDER, 'Ys_dev.csv'),  index_col=[0,1,2])
    
    if normalization == 'l_inf': 
        h5path = Path(DATA_FOLDER, 'lvl2_l_inf_normalized.h5')
    elif normalization == 'z': 
        h5path = Path(DATA_FOLDER, 'lvl2_z_normalized.h5')
    
    # read in level 2 data 
    df_train = pd.read_hdf(h5path,  key = 'train')
    df_dev = pd.read_hdf(h5path,  key = 'dev')
    df_test = pd.read_hdf(h5path,  key = 'test')

    if level == 3: 
        # pivot on the hourly measurements to make three levels
        df_train, df_dev, df_test = [ df.pivot_table(\
            index=['subject_id', 'hadm_id', 'icustay_id'], columns=['hours_in']) \
                for df in (df_train, df_dev, df_test )]
    
    # Make Pytorch Dataset class:  
    if use_full:
        df_train = pd.concat( [df_train, df_dev], axis = 0) 
        Ys_train = pd.concat([Ys_train, Ys_dev], axis = 0)
        training_data = DatasetMaker(df_train.loc[:, idx[ :, "mean" ]].values, Ys_train[target].values)
        testing_data = DatasetMaker(df_test.loc[:, idx[ :, "mean" ]].values, Ys_test[target].values)
    else:
        training_data = DatasetMaker(df_train.loc[:, idx[ :, "mean" ]].values, Ys_train[target].values)
        testing_data = DatasetMaker(df_dev.loc[:, idx[ :, "mean" ]].values, Ys_dev[target].values)
    
    return df_train,  df_test, Ys_train,  Ys_test, training_data, testing_data


def prepare_data(X, labels, batch_size):
    """
        Puts data into nf_batches
        
    Inputs:
        X: mimic data
        labels: mimic labels
        batch_size: number of samples in a batch
    Outputs:
        X, labels
    """
    labels = torch.stack([labels]*batch_size)
    X = torch.stack([X]*batch_size)
    
    return X, labels


## hyperparm search functions 
class Choice(): # pass in a list, it has one method `.rvs(n)` that returns n randomly sampled elements of list with replacement
    def __init__(self, options): self.options = options
    def rvs(self, n): return [self.options[i] for i in ss.randint(0, len(self.options)).rvs(n)]
    def __str__(self):
        return str(f'Choice({self.options})')
    def __repr__(self):
        return str(self)


class DictDist(): # pass in a dict, but values must be the Choice class. One method `rvs(n)`, creates dict running Choice.rvs(n) for each value
    def __init__(self, dict_of_rvs): self.dict_of_rvs = dict_of_rvs
    def rvs(self, n):
        a = {k: v.rvs(n) for k, v in self.dict_of_rvs.items()}
        out = []
        for i in range(n): out.append({k: vs[i] for k, vs in a.items()})
        return out
    def __repr__(self):
        return pformat(self.dict_of_rvs, width = 1)


def get_logistic_model(b, input_dim, output_dim = 1, use_gpu = False, device_id = 'cpu'):
    """ Make a Logistic model from model parameters sampled from an NF """
    lr_model = LogisticRegression(input_dim, output_dim)
    state_dict = {
        'linear.weight' : b[:-1].reshape([1, input_dim]),
        'linear.bias' : b[-1].reshape([1])
        }
    lr_model.load_state_dict(state_dict)
    if use_gpu:
        lr_model.to(device_id)
    return lr_model

    
## accuracy results functions
def get_auc(b, X_test, y_test):
    """Used to evaluate NF samples. given b (tensor of shape [X.shape[1] + 1])
        this instantiates a logistic regrssion model with b[-1] as the bias term 
        then computes the area under the ROC for the given data. 

    Args:
        b (torch.tensor): _description
        X_test (torch.tensor): _description_. Defaults to X_test.
        y_test (torch.tensor): _description_. Defaults to y_test.
        
    Returns:
        float: roc_auc 
    """
    b = b.to('cpu') # for now
    y_test = y_test.to('cpu')
    X_test = X_test.to('cpu')

    input_dim = X_test.shape[1] #  (already has the weights column removed)
    output_dim = 1
    lr_model = LogisticRegression(input_dim, output_dim)
    state_dict = {
        'linear.weight' : b[:-1].reshape([1,X_test.shape[1]]),
        'linear.bias' : b[-1].reshape([1])
        }
    lr_model.load_state_dict(state_dict)

    with torch.no_grad():
        y_pred = lr_model.forward(X_test).squeeze().detach()

    targets = y_test.detach()
    try:
        x = roc_auc_score(targets, y_pred)
    except: 
        x = np.nan

    return x


def get_metrics(b, X_test, y_test):
    """Used to evaluate NF samples. given b (tensor of shape [X.shape[1] + 1])
        this instantiates a logistic regrssion model with b[-1] as the bias term 
        then computes the area under the ROC for the given data. 

    Args:
        b (torch.tensor): _description
        X_test (torch.tensor): _description_. Defaults to X_test.
        y_test (torch.tensor): _description_. Defaults to y_test.
        
    Returns:
        float: roc_auc 
    """
    b = b.to('cpu') # for now
    y_test = y_test.to('cpu')
    X_test = X_test.to('cpu')

    input_dim = X_test.shape[1] #  (already has the weights column removed)
    output_dim = 1
    lr_model = LogisticRegression(input_dim, output_dim)
    state_dict = {
        'linear.weight' : b[:-1].reshape([1,X_test.shape[1]]),
        'linear.bias' : b[-1].reshape([1])
        }
    lr_model.load_state_dict(state_dict)

    with torch.no_grad():
        y_score = lr_model.forward(X_test).squeeze().detach()

    targets = y_test.detach()
    try:
        auc = roc_auc_score(targets, y_score)
        aps = average_precision_score(targets, y_score)
        fpr, tpr, thresholds = roc_curve(targets, y_score)
        df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresh":thresholds})
        df['dist_to_[0,1]'] = df.apply( lambda row: np.sqrt((1-row.tpr)**2 + (row.fpr)**2), axis = 1) # dist to [0,1]
        best = df[ df['dist_to_[0,1]'] == df['dist_to_[0,1]'].min()].iloc[0] # closest point on ROC to [0,1]
        thresh = best.thresh
        y_pred = (y_score>thresh) # binary predictions
        best['accuracy'] = accuracy_score(targets, y_pred)
        best['f1']  = f1_score(targets, y_pred)
        best['prec'] = precision_score(targets, y_pred)
    except: 
        auc = np.NaN # roc_auc_score(targets, y_score)
        aps = np.NaN # average_precision_score(targets, y_score)

        # get best threhold
        df = pd.DataFrame({"fpr": [np.NaN], "tpr": [np.NaN], "thresh": [np.NaN]})
        df['dist_to_[0,1]'] = np.NaN
        best = df[ df['dist_to_[0,1]'] == np.NaN] # closest point on roc to [0,1]
        # metrics at "best" threshold
        best['accuracy'] = np.NaN
        best['f1']  = np.NaN # f1_score(targets, y_pred)
        best['prec'] = np.NaN # precision_score(targets, y_pred)

    return auc, aps, best


def get_ave_auc(nf_model, X_test, y_test, sample_std, n = 1000, device = 'cpu'):
    # for now
    z_size = X_test.shape[1] + 1 # plus 1 for bias term 
    samples = random_normal_samples(n, z_size, std = sample_std, device = device)
    B, _ = nf_model.forward(samples) # l by X.shape[0] + 1 tensor, run thru NF 
    B.detach()
    return pd.Series([get_auc(b, X_test, y_test) for b in B]).mean()


# loss functions for DPSGD & nonprivate models:
def loss_bce(lr_model, X, y, lam, reduction = 'sum'): #binary cross entropy + L1 regularization. 
    """returns -\sum_i w_i [ y_i \log (\phi(x_i*beta)) + (1-y_i)\log(1-\phi(x_i*beta)) ] + c ||beta||_1

    Args:
        model (torch.nn.Model): Logistic regression model (or any binary classification model with outputs in (0,1))
        X (torch.tensor): one batch's design matrix. Shape is (batch size, number of features). each row is a feature vector
        y (torch.tensor): one batch's list of targets. Shape is (batch size, ). entries are in {0,1}
        weight (torch.tensor): one batch's list of weights. Shape is (batch size, ). Optional. Default = None
        c (float): regularization parameter. Optional. Default = 1
    Returns:
        _type_: _description_
    """
    # first compute cross entropy loss:
    y = y.squeeze()
    cel_fn = torch.nn.BCELoss(reduction = 'none' ) ## instantiates binary cross entropy loss function class, see # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    y_pred = lr_model.forward(X).squeeze() # has one extra, unused dimension we need to squeeze
    cels = cel_fn(y_pred, y) 
    if reduction == 'sum':
        cels = cels.sum()
    elif reduction == 'mean':
        cels = cels.mean()
    # next compute regularization: 
    p = list(lr_model.parameters()) # two element list. first is a list of all the non bias parameters, second element is a list of the lone bias parameter
    x = p[0].squeeze()
    x = x.abs()
    x = x.sum() # sum of abs of all but bias parameter
    w = p[1]
    w = w.abs()
    reg = x + w #sum of abs of all parameters, the l1 norm. 
    
    return (cels + lam*reg)


def loss_l2(lr_model, X, y, lam, reduction = 'sum'): # l2 loss w/ l1 reg. 
    y = y.squeeze()
    y_pred = lr_model.forward(X).squeeze()
    loss = (y_pred - y).pow(2)
    if reduction == 'sum':
        loss = loss.sum() # sum of abs of all but bias parameter
    elif reduction == 'mean':
        loss = loss.mean() # sum of abs of all but bias parameter

    p = list(lr_model.parameters())
    x = p[0].squeeze()
    x = x.abs()
    x = x.sum()
    w = p[1]
    w = w.abs()    
    reg = x + w
    return loss + lam*reg


# loss function for nf, logistic regression: 
def loss_nf_lr(model, samples, X_1, yl, lam, epsilon, s): # This manually computes the LR loss portion: 
    B, log_det_T = model(samples) # samples from the untrained NF
    y_preds = torch.sigmoid(torch.mm(X_1,B.T)) # shape is (n by l): \phi(x^t b)  for each of the n data points x\in X, and for each of the l weight vectors b\in B, 
    
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
    ## and can combine into the loss: 
#    loss = -(log_det_T + log_p_x).mean() # Reverse KL
    loss = -(log_det_T.squeeze() + log_p_x).mean() # Reverse KL
    return loss 


## nf + grud loss function
def loss_nf_grud(nf_model, wrapper, samples, X, X_last_obsv, Mask, Delta,
            labels, epsilon, s, lam = .1): 
    """
        Computes loss for normalizing flow model. Utility will be l2 loss for GRUD wrapper.
    
    Inputs:
        nf_model: a normalizing flow model
        wrapper: a GRUD wrapper model
        samples: the samples that will be pushed through the nf_model
        X: mimic data for grud
        X_last_obsv: mimic data for grud
        Mask: mimic data for grud
        Delta: mimic data for grud
        labels: mimic labels for grud
        epsilon: the privacy level
        s: the sensitivity
        lam: the regularization parameter (DEFAULT = .1) 
    Returns:
        loss
    """
    loss_fn = torch.nn.MSELoss(reduction = 'none')
    sampled_params, log_det_T = nf_model(samples) # samples from the untrained NF
#    losses = torch.mean(loss_fn(wrapper(sampled_params, X, X_last_obsv, Mask, Delta)[:,:,1], labels))
    losses = torch.sum(loss_fn(wrapper(sampled_params, X, X_last_obsv, Mask, Delta).squeeze(), labels))
                                          
    ## regularization:
    regs = sampled_params.abs()
    regs = regs.sum(axis = 1) # shape is l
    ## sum and negate to get utility: 
    util_l2 = - (losses + (lam * regs))
    ## finally, we have the target potential function: 
    log_p_x = util_l2 * epsilon/(2*s)
    ## and can combine into the loss: 
    loss = -(log_det_T + log_p_x).mean() # Reverse KL
    return loss 



# logistic regression training functions
def nonpriv_train(model, loss_fn, train_dataloader, patience, momentum, learning_rate,
                  num_epochs, lam, verbose = False, reduction = 'none'):
    opt = torch.optim.RMSprop(
        params = model.parameters(),
        lr = learning_rate,
        momentum = momentum)

    scheduler = ReduceLROnPlateau(opt, 'min', patience=patience, verbose=True) 

    losses = [] # to be populated
    steps = [] # to be populated

    for epoch in range(num_epochs): # for each epoch

        for X, y in train_dataloader: #for each batch   

            loss = loss_fn(model, X, y, lam, reduction)

            # take a step: 
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(loss)
        
        if verbose and (epoch % 100 == 0): 
            losses.append(loss.detach())
            steps.append(epoch)
            print(f"Epoch {epoch}")
            print("\tTraining Loss {}".format(loss.item()))
            # print("\tTesting Loss {}".format(loss_test.item() * len(training_data) / X_test.shape[0]))

    return model, losses


def dpsgd_train(model, loss_fn, train_dataloader, learning_rate, patience, 
                momentum, num_epochs, lam,  noise_multiplier, verbose = False):#save_state = False,  save_betas = False
    """
    Trains a private model with PRVAccountant using the given optimizer (with weights).

    Args:
        model (NN class): instantiated model
        train_dataloader (torch.DataLoader): data that the model will train on, split into batches
        epochs (int): number of iterations over entire dataset
        learning_rate (float): step size multiplier for optimization
        momentum (float): paramter for how much of last step to mix into this step
        noise_multiplier (float): how much noise to use during training
        lam = regularization constant
        verbose (bool): if True, prints the loss and epsilon (using delta) per epoch.
        save_state (bool): [functionality turned off.] if True, save the model, opacus privacy engine and optimizer
    (rest of these args are imported from config.py)
        save_betas (bool): if True it returns  the model weights after each epoch (list has len = epochs). if False it returns only the final model weights (list has length 1)
    Returns:
        train_loss: list of losses of training set each epoch
        betas: list of model weights
        ## steps_per_epoch_count
    """
    # print(f'training with epochs {epochs},  learning rate: {learning_rate}, momentum: {momentum}, nm: {noise_multiplier}')
    # Make private with Opacus
    privacy_engine = PrivacyEngine(accountant = 'prv')
    opt = torch.optim.RMSprop(
        params = model.parameters(),
        lr = learning_rate,
        momentum = momentum
        )
    priv_model, priv_opt, priv_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=opt,
            data_loader=train_dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=1.0)

    scheduler = ReduceLROnPlateau(opt, 'min', patience=patience, verbose=True) 
    # to be populated
    train_losses = []
    # steps_per_epoch_count = 0 # going to count manually during training
    total_steps = 0

    for epoch in range(num_epochs): # for each epoch
        for X, y in priv_dataloader: #for each batch   
            l = loss_fn(priv_model, X, y, lam)
            # take a step:
            priv_opt.zero_grad()
            total_steps += 1 # needed for computing privacy later
            l.backward()
            priv_opt.step()
            scheduler.step(l)

        # this allows us to compute privacy later with a variety of deltas if we so choose
        train_losses.append(l.item())  #
        
        if verbose and epoch % 1 == 0:
            # now record losses on train set 
            print(f"Epoch {epoch}")
            print("\tTraining Loss {}".format(l.item()))


    return priv_model, privacy_engine, total_steps, train_losses#, steps_per_epoch_count


def nf_train(model, X_1, yl, epsilon, s, lam, z_size, sample_std, learning_rate,
             momentum, epochs, batch_size, patience, device_type, verbose = False):
    """
    Trains model with RMSprop using given lr, momentum, epochs. Uses ReduceLROnPlateau scheduler. 
    Args:
        model (NN class): instantiated model    
        X_1 (torch.tensor): X_train with an extra column of ones (for bias term)
        yl (torch.tensor): y_training * batchsize
        wl (): weight-related argument for the given loss function:
            for loss_nf_l2 this is wl = w_train * batchsize,
            for loss_nf_cel thi is cel_fn = BCELoss function instantiated with w_train 
        epsilon (float): privacy parameter
        verbose (bool): if True, prints the loss per epoch. 
        c (float): regularization parameter. 
        sample_std (float): standard deviation of NF model's base normal distribution (a hyperparameter)
        learning_rate (float): step size multiplier for optimization
        momentum (float): parameter for how much of last step to mix into this step
        epochs (int): number of mini batches to use in training
        batch_size (int): number of samples to use in each mini batch
        patience (int): how long to wait before applying scheduler
    Returns:
        tuple: losses, model (trained normalizing flow model)
    """
    global loss_nf_lr
    
    opt = torch.optim.RMSprop(\
        params = model.parameters(),\
        lr = learning_rate,\
        momentum = momentum)
    scheduler = ReduceLROnPlateau(opt, 'min', patience=patience, verbose=verbose) 
    
    for _ in range(epochs):
        samples = random_normal_samples(batch_size, z_size, std = sample_std, device = device_type)
        loss = loss_nf_lr(model, samples, X_1, yl, lam, epsilon, s)
        # take a step: 
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step(loss)

    return model


## grud training function..? is this used? 
def nf_train_grud(nf_model, model_wrapper, train_dataloader, opt, scheduler, 
          epochs, batch_size, z_size, sample_std,
          lam, epsilon, s, device_type, skip = 100, verbose = False):
    """
        Trains the normalizing flow model for grud

    Inputs:
        nf_model: normalizing flow model
        model_wrapper: grud model wrapper
        train_dataloader: data loader for training
        opt: optimizer
        scheduler: decreasing learning rate
        epochs: how many times over the training data
        batch_size: number of samples to push through normalizing flow
        z_size: number of parameters in wrapper model
        sample_std: variance of sample that will be pushed through nf model
        lam: regularizaiton parameter
        device_type: either a gpu id or 'cpu'
        skip: how often to save the loss
        verbose: whether to print (DEFAULT False)
    """
    steps = []
    losses = [] 

    n = 0
    for epoch in range(epochs):
        for X, y in train_dataloader:
            X = X.to(device_type)
            y = y.to(device_type)
            X, y = prepare_data(X, y, batch_size)
            # TODO: should this go before or after the data iterator loop?
            samples = random_normal_samples(batch_size, z_size, std = sample_std, device = device_type)
            # put batch on gpu (because we were running out of memory)
            loss = loss_nf_grud(nf_model, model_wrapper, samples, 
                           X, y,
                           epsilon, s, lam = lam)
            # take a step: 
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(loss)

        # Store/print loss values: 
        if epoch % skip == 0: 
            steps.append(epoch)
            losses.append(loss.item())
            if verbose: print(f'Epoch {epoch}\t loss {losses[-1]}')
    return nf_model, steps, losses


## dpsgd utils: 
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


# mia utils

def logit_confs(model, X, y):
    y_preds = model(X).squeeze()
    bce = torch.nn.BCELoss(reduction = 'none')  
    loss = -bce(y_preds, y)
    confs = torch.exp(loss)
    return loss - torch.log(1-confs) # should be numerical stable...

# this is implemented assuming binary classification
def logit_confs_stable(model, X, y):
    y_preds = model(X).squeeze()
    # get best threhold 
    bce = torch.nn.BCELoss(reduction = 'none')  
    loss = -bce(y_preds, y)
    loss_y_prime = -bce(y_preds, 1-y) # flip the labels to show the loss for the remaining class
    # log(exp(loss)) - log(1-exp(loss))
    # = loss - log(exp(loss_y_prime))
    # = loss - loss_y_prime 
    return loss - loss_y_prime # probably more stable...

def logit_confs_lr_hinge(model, X, y):
    z_y = model.linear(X).squeeze()
    # section 4A hinge loss 
    # l(x,y) = z_y - max_{y' != y} z_{y'}
    # the output of our linear layer is 1-d so we compute
    # l(x,y) = z_y - (1- z_y) = 2z_y - 1
    return 2*z_y-1


def score(model, loss_fn, X, y):
    y_preds = model(X).squeeze()

    return loss_fn(y_preds, y)




# def loss_nf(nf_model, wrapper, samples, X,
#             labels, epsilon, s, lam): 
#     """
#         Computes loss for normalizing flow model. Utility will be l2 loss for GRUD wrapper.
    
#     Inputs:
#         nf_model: a normalizing flow model
#         wrapper: a GRUD wrapper model
#         samples: the samples that will be pushed through the nf_model
#         X: mimic data for grud
#         labels: mimic labels for grud
#         epsilon: the privacy level
#         s: the sensitivity
#         lam: the regularization parameter (DEFAULT = .1) 
#     Returns:
#         loss
#     """
#     wrapper_loss_l2 = nn.MSELoss(reduction = 'none')
#     sampled_params, log_det_T = nf_model(samples) # samples from the untrained NF
# #    losses = torch.mean(loss_fn(wrapper(sampled_params, X, X_last_obsv, Mask, Delta)[:,:,1], labels))
#     losses = torch.sum(wrapper_loss_l2(wrapper(sampled_params, X).squeeze(), labels))                          
#     ## regularization:
#     regs = sampled_params.abs()
#     regs = regs.sum(axis = 1) # shape is l
#     ## sum and negate to get utility: 
#     util_l2 = - (losses + (lam * regs))
#     ## finally, we have the target potential function: 
#     log_p_x = util_l2 * epsilon/(2*s)
#     ## and can combine into the loss: 
#     loss = -(log_det_T + log_p_x).mean() # Reverse KL
#     return loss 
