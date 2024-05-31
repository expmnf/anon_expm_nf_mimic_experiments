# file is from MIMIC Extract Paper 
import copy, math, os, pickle, time, pandas as pd, numpy as np, scipy.stats as ss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

from opacus import PrivacyEngine, GradSampleModule
import torch, torch.utils.data as utils, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau

use_gpu = True

def to_3D_tensor(df):
    # from https://github.com/MLforHealth/MIMIC_Extract/blob/master/notebooks/mmd_grud_utils.py
    idx = pd.IndexSlice
    return np.dstack(list(df.loc[idx[:,:,:,i], :].values for i in sorted(set(df.index.get_level_values('hours_in')))))

def prepare_dataloader(df, Ys, batch_size, shuffle=True, label_type = np.float32):
    # from https://github.com/MLforHealth/MIMIC_Extract/blob/master/notebooks/mmd_grud_utils.py
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
def create_dataloader(X, label, batch_size, shuffle=True, label_type = torch.float32, drop_last = True):
    l = label.type(label_type)
    dataset = utils.TensorDataset(X, l)
    
    return utils.DataLoader(dataset, int(batch_size), shuffle=shuffle, drop_last = drop_last)



class FilterLinear(nn.Module):
    # from https://github.com/MLforHealth/MIMIC_Extract/blob/master/notebooks/mmd_grud_utils.py
    def __init__(self, in_features, out_features, filter_square_matrix, device_id, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        assert in_features > 1 and out_features > 1, "Passing in nonsense sizes"
        
#        use_gpu = False #torch.cuda.is_available()
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
 
class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, X_mean, device_id = None, 
                 batch_size = 0, use_bn = True, 
                 apply_sigmoid = False, dropout_p = .5):
        """
        With minor modifications from https://github.com/zhiyongc/GRU-D/
        And further modifications from https://github.com/MLforHealth/MIMIC_Extract

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
            use_bn: Apply a bn on the final layer (DEFAULT True)
            apply_sigmoid: Apply a sigmoid to the final prediction layer (use with l2 loss) (DEFAULT False)
            dropout_p: Probability of dropout for Dropout layer (DEFAULT = .5)
        """
        
        super(GRUD, self).__init__()
        
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.device_id = device_id
        self.use_bn = use_bn
        self.apply_sigmoid = apply_sigmoid
        
        if use_gpu:
            self.identity = torch.eye(input_size).to(self.device_id)
            self.X_mean = Variable(torch.Tensor(X_mean).to(self.device_id))
        else:
            self.identity = torch.eye(input_size)
            self.X_mean = Variable(torch.Tensor(X_mean))
        
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # Wz, Uz are part of the same network. the bias is bz
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # Wr, Ur are part of the same network. the bias is br
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # W, U are part of the same network. the bias is b
        
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity, self.device_id)
        
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size) 
        
        self.fc = nn.Linear(self.hidden_size, 1) # a probability score
        self.drop=nn.Dropout(p=dropout_p, inplace=False)
        if use_bn:
            self.bn= torch.nn.BatchNorm1d(self.hidden_size, eps=1e-05, momentum=0.1, affine=True)
        
    def step(self, x, x_last_obsv, x_mean, h, mask, delta, zeros, zeros_h):
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
        
        gamma_x_l_delta = self.gamma_x_l(delta)
        delta_x = torch.exp(-torch.max(zeros, gamma_x_l_delta)) #exponentiated negative rectifier
        
        gamma_h_l_delta = self.gamma_h_l(delta)
        delta_h = torch.exp(-torch.max(zeros_h, gamma_h_l_delta)) #self.zeros became self.zeros_h to accomodate hidden size != input size

        x_mean = x_mean.repeat(batch_size, 1)
        
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h
        
        combined = torch.cat((x, h, mask), 1)
        z = torch.sigmoid(self.zl(combined)) #sigmoid(W_z*x_t + U_z*h_{t-1} + V_z*m_t + bz)
        r = torch.sigmoid(self.rl(combined)) #sigmoid(W_r*x_t + U_r*h_{t-1} + V_r*m_t + br)
        combined_new = torch.cat((x, r*h, mask), 1)
        hl = self.hl(combined_new)
        h_tilde = torch.tanh(hl) #tanh(W*x_t +U(r_t*h_{t-1}) + V*m_t) + b
        h = (1 - z) * h + z * h_tilde
        
        return h
    
    def forward(self, X, X_last_obsv, Mask, Delta):
        
        batch_size = X.size(0)
        step_size = X.size(1) # num timepoints
        spatial_size = X.size(2) # num features
        
        Hidden_State = self.initHidden(batch_size)

        # initialize zeros here because batch size can change with dpsgd
        zeros = Variable(torch.zeros(batch_size, self.delta_size, dtype = torch.float32).to(self.device_id))
        zeros_h = Variable(torch.zeros(batch_size, self.hidden_size, dtype = torch.float32).to(self.device_id))
        
        for i in range(step_size):
            Hidden_State = self.step(
                torch.squeeze(X[:,i:i+1,:], 1),
                torch.squeeze(X_last_obsv[:,i:i+1,:], 1),
                torch.squeeze(self.X_mean[:,i:i+1,:], 1),
                Hidden_State,
                torch.squeeze(Mask[:,i:i+1,:], 1),
                torch.squeeze(Delta[:,i:i+1,:], 1),
                zeros,
                zeros_h
            )

        # we want to predict a binary outcome
        #Apply 50% dropout and batch norm here
        if self.use_bn and self.apply_sigmoid:
            return torch.sigmoid(self.fc(self.bn(self.drop(Hidden_State))))
        elif self.use_bn and not self.apply_sigmoid:
            return self.fc(self.bn(self.drop(Hidden_State)))
        elif not self.use_bn and self.apply_sigmoid:
            return torch.sigmoid(self.fc(self.drop(Hidden_State)))
        elif not self.use_bn and not self.apply_sigmoid:
            return self.fc(self.drop(Hidden_State))
                
    
    def initHidden(self, batch_size):
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).to(self.device_id))
            return Hidden_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State


def Train_Model(model, loss_fn, train_dataloader, num_epochs = 300, patience = 1000, 
                learning_rate=1e-3, batch_size=None):
    # with modifications from https://github.com/MLforHealth/MIMIC_Extract.git
    """
    Inputs:
        model: a GRUD model
        loss_fn: the loss function to use (bce, l2)
        train_dataloader: training data
        num_epochs: number of times over the training data
        patience: used for decreasing learning rate
        learing_rate: the step size for each step of training
        batch_size: size of a batch
        
    Returns:
        best_model
        losses_train 
        losses_epochs_train
    """
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose = True) 

    losses_train = []
    losses_epochs_train = []

    X_mean = model.X_mean
    for epoch in range(num_epochs):
        losses_epoch_train = []

        for X, labels in train_dataloader:
            mask = X[:, np.arange(0, X.shape[1], 3), :]
            measurement = X[:, np.arange(1, X.shape[1], 3), :]
            time_ = X[:, np.arange(2, X.shape[1], 3), :]

            mask = torch.transpose(mask, 1, 2)
            measurement = torch.transpose(measurement, 1, 2)
            time_ = torch.transpose(time_, 1, 2)
#            measurement_last_obsv = measurement
            m_shape = measurement.shape[0]
            # we delete last column and prepend mean so that the last observed is used
            measurement_last_obsv = measurement[:, 0:measurement.shape[1]-1, :]
            measurement_last_obsv = torch.cat((torch.stack([X_mean[:, 0, :]]*m_shape), 
                                               measurement_last_obsv), dim = 1)

            assert measurement.size()[0] == batch_size, "Batch Size doesn't match! %s" % str(measurement.size())

            convert_to_tensor = lambda x: torch.autograd.Variable(x)
            X, X_last_obsv, Mask, Delta, labels = map(convert_to_tensor, 
                                                 [measurement, 
                                                  measurement_last_obsv,
                                                  mask,
                                                  time_,
                                                  labels])
        
            model.zero_grad()

            prediction = model(X, X_last_obsv, Mask, Delta)

            loss_train = loss_fn(torch.squeeze(prediction), torch.squeeze(labels))
            with torch.no_grad():
                losses_train.append(loss_train.item())
                losses_epoch_train.append(loss_train.item())

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step(loss_train)


        avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
        losses_epochs_train.append(avg_losses_epoch_train)

    return model, [losses_train, losses_epochs_train]
 
def Train_Model_DPSGD(pre_model, loss_fn, pre_train_dataloader, noise_multiplier, 
                      max_grad_norm = 1, num_epochs = 300, patience = 1000, 
                      learning_rate=1e-3, batch_size=None):
    """
    Inputs:
        pre_model: a GRUD model
        loss_fn: the loss function to use
        pre_train_dataloader: training data
        noise_multiplier: the noise multiplier for dpsgd
        max_grad_norm: the max norm for gradient in dpsgd
        num_epochs: number of times over the training data
        patience: used for decreasing learning rate
        min_delta: if the loss stays within this value on the next step stop early
        batch_size: size of a batch
        
    Returns:
        best_model
        losses_train 
        losses_epochs_train
    """
    pre_opt = torch.optim.Adam(pre_model.parameters(), lr = learning_rate)

    # make private
    privacy_engine = PrivacyEngine(accountant = 'prv')
    priv_model, priv_opt, priv_train_dataloader = privacy_engine.make_private(
        module=pre_model,
        optimizer=pre_opt,
        data_loader=pre_train_dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    scheduler = ReduceLROnPlateau(priv_opt, 'min', patience=patience, verbose = True) 

#    losses_train = []
#    losses_epochs_train = []
    niter_per_epoch = 0

    # BE CAREFUL! The mean should be computed privately.
    X_mean = priv_model._module.X_mean
    for epoch in range(num_epochs):
#        losses_epoch_train = []

        for X, labels in priv_train_dataloader:
            if epoch == 0:
                niter_per_epoch += 1 # needed to compute epsilon later if we want to
            mask = X[:, np.arange(0, X.shape[1], 3), :]
            measurement = X[:, np.arange(1, X.shape[1], 3), :]
            time_ = X[:, np.arange(2, X.shape[1], 3), :]

            mask = torch.transpose(mask, 1, 2)
            measurement = torch.transpose(measurement, 1, 2)
            time_ = torch.transpose(time_, 1, 2)
#            measurement_last_obsv = measurement
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
        
            priv_model.zero_grad()

            prediction = priv_model(X, X_last_obsv, Mask, Delta)

            loss_train = loss_fn(torch.squeeze(prediction), torch.squeeze(labels))
#            with torch.no_grad():
#                losses_train.append(loss_train.item())
#                losses_epoch_train.append(loss_train.item())

            priv_opt.zero_grad()
            loss_train.backward()
            priv_opt.step()
            scheduler.step(loss_train)

#        avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
#        losses_epochs_train.append(avg_losses_epoch_train)

    return priv_model, niter_per_epoch, privacy_engine#, [losses_train, losses_epochs_train]
 

# this is implemented assuming binary classification
def logit_confs_stable(grud_model, dataloader):
    probabilities, labels = predict_proba(grud_model, dataloader, numpy = False)
    y_preds = torch.concatenate(probabilities).squeeze()
    y  = torch.concatenate(labels).squeeze()
    # get best threhold 
#    bce = torch.nn.BCELoss(reduction = 'none')  
    bce = torch.nn.BCEWithLogitsLoss(reduction = 'none')
    loss = -bce(y_preds, y)
    loss_y_prime = -bce(y_preds, 1-y) # flip the labels to show the loss for the remaining class
    # log(exp(loss)) - log(1-exp(loss))
    # = loss - log(exp(loss_y_prime))
    # = loss - loss_y_prime 
    return loss - loss_y_prime # probably more stable...

def logit_confs_grud_hinge(grud_model, dataloader):
    # section 4A hinge loss 
    grud_model.apply_sigmoid = False # get output before sigmoid is applied
    probabilities, labels = predict_proba(grud_model, dataloader, numpy = False)
    z_y = torch.concatenate(probabilities).squeeze()

    return 2*z_y-1


def score(grud_model, loss_fn, dataloader):
    probabilities, labels = predict_proba(grud_model, dataloader, numpy = False)
    y_preds = torch.concatenate(probabilities).squeeze()
    y  = torch.concatenate(labels).squeeze()

    return loss_fn(y_preds, y)




def predict_proba(model, dataloader, numpy = True):
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
    if type(model) == GradSampleModule:
        X_mean = model._module.X_mean
    else:
        X_mean = model.X_mean
    
    for X, label in dataloader:
        mask        = X[:, np.arange(0, X.shape[1], 3), :]
        measurement = X[:, np.arange(1, X.shape[1], 3), :]
        time_       = X[:, np.arange(2, X.shape[1], 3), :]

        mask = torch.transpose(mask, 1, 2)
        measurement = torch.transpose(measurement, 1, 2)
        time_ = torch.transpose(time_, 1, 2)
#        measurement_last_obsv = measurement            
        m_shape = measurement.shape[0]
        # we delete last column and prepend mean so that the last observed is used
        measurement_last_obsv = measurement[:, 0:measurement.shape[1]-1, :]
        measurement_last_obsv = torch.cat((torch.stack([X_mean[:, 0, :]]*m_shape), 
                                           measurement_last_obsv), dim = 1)

        convert_to_tensor=lambda x: Variable(x)
        X, X_last_obsv, Mask, Delta, label  = map(convert_to_tensor, [measurement, measurement_last_obsv, mask, time_, label])

        prob = model(X, X_last_obsv, Mask, Delta)
        
        if numpy:
            probabilities.append(prob.detach().cpu().data.numpy())
            labels.append(label.detach().cpu().data.numpy())
        else:
            probabilities.append(prob.detach())
            labels.append(label.detach())

    return probabilities, labels
