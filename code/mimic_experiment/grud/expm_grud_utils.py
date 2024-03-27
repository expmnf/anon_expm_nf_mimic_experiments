import copy, math, sys, os, pickle, time, pandas as pd, numpy as np, scipy.stats as ss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
from pathlib import Path 

import torch, torch.utils.data as utils, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

sys.path.append(Path(".", "code").as_posix())
sys.path.append(Path(".", "code", "mimic_experiment").as_posix())
from normalizing_flows import * 
from experiment_utils import *

def to_3D_tensor(df):
    # from https://github.com/MLforHealth/MIMIC_Extract.git
    idx = pd.IndexSlice
    return np.dstack(list(df.loc[idx[:,:,:,i], :].values for i in sorted(set(df.index.get_level_values('hours_in')))))

def prepare_dataloader(df, Ys, batch_size, shuffle=True, label_type = np.float32):
    # from https://github.com/MLforHealth/MIMIC_Extract.git
    """
    dfs = (df_train, df_dev, df_test).
    df_* = (subject, hadm, icustay, hours_in) X (level2, agg fn \ni {mask, mean, time})
    Ys_series = (subject, hadm, icustay) => label.
    """
    X     = torch.from_numpy(to_3D_tensor(df).astype(np.float32))
    label = torch.from_numpy(Ys.values.astype(label_type))
    dataset = utils.TensorDataset(X, label)
    
    return utils.DataLoader(dataset, int(batch_size), shuffle=shuffle, drop_last = True)

# same as above but doesn't use a dataframe
def create_dataloader(X, label, batch_size, shuffle=True):
    dataset = utils.TensorDataset(X, label)
    
    return utils.DataLoader(dataset, int(batch_size), shuffle=shuffle, drop_last = True)


class FilterLinear(nn.Module):
    # from https://github.com/MLforHealth/MIMIC_Extract.git
    def __init__(self, in_features, out_features, filter_square_matrix, device_id, use_gpu, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        assert in_features > 1 and out_features > 1, "Passing in nonsense sizes"
        
        self.filter_square_matrix = None
        if use_gpu: self.filter_square_matrix = Variable(filter_square_matrix.to(device_id), requires_grad=False)
        else:       self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        if bias: self.bias = Parameter(torch.Tensor(out_features))
        else:    self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None: self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return F.linear(
            x,
            self.filter_square_matrix.mul(self.weight),
            self.bias
        )

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    
class ParallelGRUDWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, X_mean, device_id, use_gpu,
                 dropout_p = .5, batch_size = 0, apply_sigmoid = True):
        """
        An implementation for ExpNF method that runs GRUD "in parallel" on the sampled parameters

        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        
        Implemented based on the paper: 
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }
        
        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the historical input data
            device_id: if running on the gpu, provide the device number
            batch_size: number of elements in a batch
            apply_sigmoid: Apply a sigmoid to the final prediction layer (DEFAULT True)
            dropout_p: Probability of dropout for Dropout layer (DEFAULT = .5)

        """
        
        super(ParallelGRUDWrapper, self).__init__()
        
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.device_id = device_id
        self.use_gpu = use_gpu
        self.apply_sigmoid = apply_sigmoid

        
        if use_gpu:
            self.identity = torch.eye(input_size).to(self.device_id)
            self.zeros = Variable(torch.zeros(batch_size, input_size, dtype = torch.float32).to(self.device_id))
            self.zeros_h = Variable(torch.zeros(batch_size, self.hidden_size, dtype = torch.float32).to(self.device_id))
            self.X_mean = Variable(torch.Tensor(X_mean).to(self.device_id))
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(batch_size, input_size))
            self.zeros_h = Variable(torch.zeros(batch_size, self.hidden_size))
            self.X_mean = Variable(torch.Tensor(X_mean))
        
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # Wz, Uz are part of the same network. the bias is bz
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # Wr, Ur are part of the same network. the bias is br
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # W, U are part of the same network. the bias is b
        
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity, self.device_id, use_gpu)
        
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size) 
        
        
        self.fc = nn.Linear(self.hidden_size, 1)
        self.drop=nn.Dropout(p=dropout_p, inplace=False)


        self.nparams, self.param_range_and_size = self._get_parameter_attr()
        
    def _get_parameter_attr(self):
        total_params = 0
        range_and_size = dict()
        prev_layer = ""
        start = 0
        for name, param in self.named_parameters():
            layer, sublayer = name.split(".")
            s = math.prod(param.size())
            # we don't want to repeat for bias and weight, etc.
            
            total_params += s
            end = s + start
            if layer in range_and_size:
                range_and_size[layer].update({sublayer : (range(start, end), list(param.shape))})
            else:
                range_and_size[layer] = {sublayer: (range(start, end), list(param.shape))}
            prev_layer = layer
            start = end
        return total_params, range_and_size

    def call_linear_layer(self, layer_name, all_params, x):
        param_info = self.param_range_and_size[layer_name]
        w_range = param_info['weight'][0]
        w_size = param_info['weight'][1]
        ws = all_params[:, w_range].reshape(-1, w_size[0], w_size[1])
        b_range = param_info['bias'][0]
        b_size = param_info['bias'][1]
        bs = all_params[:, b_range]
        return torch.bmm(x, ws.permute(0,2,1)) + bs.unsqueeze(1)

    def call_filter_linear_layer(self, layer_name, all_params, x):
        param_info = self.param_range_and_size[layer_name]
        w_range = param_info['weight'][0]
        w_size = param_info['weight'][1]
        ws = all_params[:, w_range].reshape(-1, w_size[0], w_size[1])
        filtered = getattr(self, layer_name).filter_square_matrix.mul(ws) # filter weights
        b_range = param_info['bias'][0]
        b_size = param_info['bias'][1]
        bs = all_params[:, b_range]
        return torch.bmm(x, filtered.permute(0,2,1)) + bs.unsqueeze(1)

    def step(self, params, x, x_last_obsv, x_mean, h, mask, delta):
        """
        Inputs:
            x: input tensor
            x_last_obsv: input tensor with forward fill applied
            x_mean: the mean of each feature
            h: the hidden state of the network
            mask: the mask of whether or not the current value is observed
            delta: the tensor indicating the number of steps since the last time a feature was observed.
            
        Returns:
            h: the updated hidden state of the network
        """
        
        batch_size = x.size()[0]
        dim_size = x.size()[1]
        
        # gamma_x_l_delta = self.gamma_x_l(delta)
        gamma_x_l_delta = self.call_filter_linear_layer('gamma_x_l', params, delta)
        delta_x = torch.exp(-torch.max(self.zeros, gamma_x_l_delta)) #exponentiated negative rectifier
        
        # gamma_h_l_delta = self.gamma_h_l(delta)
        gamma_h_l_delta = self.call_linear_layer('gamma_h_l', params, delta)
        delta_h = torch.exp(-torch.max(self.zeros_h, gamma_h_l_delta)) #self.zeros became self.zeros_h to accomodate hidden size != input size

        x_mean = x_mean.repeat(batch_size, 1)
        
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h
        
        # combined = torch.cat((x, h, mask), 1)
        combined = torch.cat((x, h, mask), 2).reshape(x.shape[0], x.shape[1], -1)
        # z = torch.sigmoid(self.zl(combined)) #sigmoid(W_z*x_t + U_z*h_{t-1} + V_z*m_t + bz)
        z = torch.sigmoid(self.call_linear_layer('zl', params, combined))
        # r = torch.sigmoid(self.rl(combined)) #sigmoid(W_r*x_t + U_r*h_{t-1} + V_r*m_t + br)
        r = torch.sigmoid(self.call_linear_layer('rl', params,  combined))
        # combined_new = torch.cat((x, r*h, mask), 1)
        combined_new = torch.cat((x, r*h, mask), 2).reshape(x.shape[0], x.shape[1], -1)
        # hl = self.hl(combined_new)
        hl = self.call_linear_layer('hl', params, combined_new)

        h_tilde = torch.tanh(hl) #tanh(W*x_t +U(r_t*h_{t-1}) + V*m_t) + b
        h = (1 - z) * h + z * h_tilde
        
        return h
    
    def forward(self, params, X, X_last_obsv, Mask, Delta):
        
        batch_size = X.size(0)
        step_size = X.size(1) # num timepoints
        spatial_size = X.size(2) # num features
        
        Hidden_State = self.initHidden(batch_size)
        
        nf_batch = params.shape[0]
        for i in range(step_size):
            Hidden_State = self.step(params, 
                torch.squeeze(X[:,i:i+1,:], 1),
                torch.squeeze(X_last_obsv[:,i:i+1,:], 1),
                torch.squeeze(self.X_mean[:,i:i+1,:], 1),
                Hidden_State,
                torch.squeeze(Mask[:,:,i:i+1,:], 2),
                torch.squeeze(Delta[:,:,i:i+1,:], 2),
            )

        # we want to predict a binary outcome
        if self.apply_sigmoid:
            return torch.sigmoid(self.call_linear_layer('fc', params, self.drop(Hidden_State)))
        else:
            return self.call_linear_layer('fc', params, self.drop(Hidden_State))
     
               
    def initHidden(self, batch_size):
        if self.use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).to(self.device_id))
            return Hidden_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State

    

def predict_proba(model, params, dataloader, device, use_gpu):
    # with modifications from https://github.com/MLforHealth/MIMIC_Extract.git
    """
    Input:
        model: GRU-D model
        test_dataloader: containing batches of measurement, measurement_last_obsv, mask, time_, labels
    Returns:
        predictions: size[num_samples, 2]
        labels: size[num_samples]
    """
    model.eval()
    
    probabilities = []
    labels        = []
    ethnicities   = []
    genders       = []
    
    if params.ndim == 1:
        all_params = params.unsqueeze(0)
    else:
        all_params = params
    nf_batch_size = all_params.shape[0]
    
    X_mean = model.X_mean
    for X, label in dataloader:
        # put batch on gpu (because we were running out of memory)
        labels.append(label.detach().cpu().data.numpy())
        if use_gpu:
            X = X.to(device)
            label = label.to(device)

        mask        = X[:, np.arange(0, X.shape[1], 3), :]
        measurement = X[:, np.arange(1, X.shape[1], 3), :]
        time_       = X[:, np.arange(2, X.shape[1], 3), :]

        mask = torch.transpose(mask, 1, 2)
        measurement = torch.transpose(measurement, 1, 2)
        time_ = torch.transpose(time_, 1, 2)
        m_shape = measurement.shape[0]
        # we delete last column and prepend mean so that the last observed is used
        measurement_last_obsv = measurement[:, 0:measurement.shape[1]-1, :]
        measurement_last_obsv = torch.cat((torch.stack([X_mean[:, 0, :]]*m_shape), 
                                           measurement_last_obsv), dim = 1)

        convert_to_tensor=lambda x: Variable(x)
        X, X_last_obsv, Mask, Delta, label  = map(convert_to_tensor, [measurement, measurement_last_obsv, mask, time_, label])

        Delta = torch.stack([Delta]*nf_batch_size)
        Mask = torch.stack([Mask]*nf_batch_size)
        
        prob = model(all_params, X, X_last_obsv, Mask, Delta)
        
        probabilities.append(prob.detach().cpu().data.numpy())

    return probabilities, labels


def get_auc_with_wrapper(grud_model, params, dataloader, device, use_gpu):
    probabilities_dev, labels_dev = predict_proba(grud_model, params, dataloader, device, use_gpu)
    targets  = np.concatenate(labels_dev)
    y_scores = np.concatenate(probabilities_dev, axis = 1)
    return [roc_auc_score(targets, y_score) for y_score in y_scores]


def get_metrics(grud_model, params, dataloader):
    probabilities_dev, labels_dev = predict_proba(grud_model, test_dataloader)
    y_score = np.concatenate(probabilities_dev)
    y_pred  = np.argmax(probabilities_dev)
    targets  = np.concatenate(labels_dev)

    auc = roc_auc_score(targets, y_score)
    aps = average_precision_score(targets, y_score)
    
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

    return auc, aps, best


def prepare_data(X, X_mean, labels, batch_size):
    """
        Prepares data for GRUD model
        
    Inputs:
        X: mimic data
        labels: mimic labels
        batch_size: number of samples in a batch
    Outputs:
        X, X_last_obsv, Mask, Delta, labels
    """
    mask = X[:, np.arange(0, X.shape[1], 3), :]
    measurement = X[:, np.arange(1, X.shape[1], 3), :]
    time_ = X[:, np.arange(2, X.shape[1], 3), :]

    mask = torch.transpose(mask, 1, 2)
    measurement = torch.transpose(measurement, 1, 2)
    time_ = torch.transpose(time_, 1, 2)

    m_shape = measurement.shape[0]
    # we delete last column and prepend mean so that the last observed is used
    measurement_last_obsv = measurement[:, 0:measurement.shape[1]-1, :]
    measurement_last_obsv = torch.cat((torch.stack([X_mean[:, 0, :]]*m_shape), 
                                       measurement_last_obsv), dim = 1)


    convert_to_tensor = lambda x: torch.autograd.Variable(x)
    X, X_last_obsv, Mask, Delta, labels = map(convert_to_tensor, 
                                         [measurement, 
                                          measurement_last_obsv,
                                          mask,
                                          time_,
                                          labels])
    # get Delta and Mask in nf_batched format
    Delta = torch.stack([Delta]*batch_size)
    Mask = torch.stack([Mask]*batch_size)
    labels = torch.stack([labels]*batch_size)
    
    return X, X_last_obsv, Mask, Delta, labels


def train(nf_model, model_wrapper, train_dataloader, opt, scheduler, 
          epochs, batch_size, z_size, sample_std, use_gpu,
          lam, epsilon, s, device_type, verbose = False):
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
#    steps = []
#    losses = [] 

    n = 0
    #start = timer()
    for epoch in range(epochs):
        for X, labels in train_dataloader:
            # TODO: should this go before or after the data iterator loop?
            samples = random_normal_samples(batch_size, z_size, std = sample_std, device = device_type)
            # put batch on gpu (because we were running out of memory)
            if use_gpu:
                X = X.to(device_type)
                labels = labels.to(device_type)
            
            loss = loss_nf_grud(nf_model, model_wrapper, samples, 
                           *prepare_data(X, model_wrapper.X_mean, labels, batch_size), 
                           epsilon, s, lam = lam)
            # take a step: 
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(loss)

    return nf_model#, steps, losses


def get_dataloaders(X_train, label_train, X_test, label_test, data_batch_size):
    """
        Creates dataloaders for training 
    """
    all_train_subjects = list(np.random.permutation(range(len(label_train))))
    train_subjects = all_train_subjects#[0:999]
    # just to match what was done before
    train_subjects.sort()
    X_train = X_train[train_subjects]
    label_train = label_train[train_subjects]
   
    train_dataloader = create_dataloader(X_train, label_train, batch_size=data_batch_size)
    test_dataloader = create_dataloader(X_test, label_test, batch_size=data_batch_size)
    return train_dataloader, test_dataloader

def create_grud_model(h, X_mean, data_batch_size, device_type, use_gpu):
    """
        Creates GRUD model
    """
 ## Set up GRUD wrapper
    base_params = {'X_mean': X_mean, 'input_size': X_mean.shape[2], 'device_id': device_type, 
                   'batch_size': data_batch_size, 'hidden_size': h['hidden_size'], 'use_gpu': use_gpu}
    grud_wrapper = ParallelGRUDWrapper(**base_params)
    return grud_wrapper

def evaluate_nf(nf_model, model_wrapper, h, test_dataloader, z_size, sample_std, device_type, use_gpu):
    """
        Evaluates normalizing flow model by first pushing samples through normalizing flow model 
        and then using those outputs as parameters to the grud model
    """
    with torch.no_grad():
        samples = random_normal_samples(1000, z_size, sample_std, device = device_type) # l by X.shape[0] + 1 tensor, {u_i: i = 1,..,l} sampled from base. 
        B, _ = nf_model.forward(samples) 
        B.detach()
        # TODO: how should we save the model? 
        #h['betas'] = B.tolist() I'm not sure we should save all these... but we probably should save the flow?
        h['aucs'] = get_auc_with_wrapper(model_wrapper, B, test_dataloader, device = device_type, use_gpu = use_gpu) # puts AUC for each sampled pararamter set into dict h
        x = pd.Series(h['aucs'])
        h['auc_ave']= x.mean()  # stores mean AUC across all samples 

    return h



