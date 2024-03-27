import os, importlib
# %% import needed modules:  
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from pathlib import Path 
from torch.multiprocessing import Pool, cpu_count, set_start_method, current_process
from multiprocessing.pool import ThreadPool
from itertools import chain
from time import sleep

# from matplotlib import pyplot as plt
import logging, copy, warnings, torch, sys, itertools
logging.getLogger('matplotlib.font_manager').disabled = True # turns off warnings about fonts. 
warnings.simplefilter(action='ignore', category=FutureWarning) #supress future warnings
# from sklearn.metrics import RocCurveDisplay, average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(Path(".", "code").as_posix())
from data_utils import *
# from regression_models import LogisticRegression
from normalizing_flows import * 
sys.path.append(Path(".", "code", "mimic_experiment").as_posix())
from config import * 
from experiment_utils import *
import expm_grud_utils
from expm_grud_utils import *
import mmd_grud_utils
from mmd_grud_utils import *


#### GPU SET UP

use_gpu = True
# for cdat machine, run `watch -n .5 nvidia-smi` to see available gpus
device_ids = [0] 
proc_per_gpu = 3
processes = proc_per_gpu*len(device_ids) # each process will be on one gpu

# because pytorch and mig don't play nicely we can only use the gpus 
# that aren't those so we can use multiprocessing.
if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([f'{i}' for i in device_ids])

### for gpu
def get_gpu_id(gpu_ids = device_ids):
    return 0

h = {'hidden_size': 10,
    'learning_rate': 0.004592623182789133,
    'num_epochs': 10,
    'patience': 6,
    'batch_size': 53,
    'dropout_p' : .5}

target = "mort_icu"

reload = True
if reload:
    df_train, df_dev, df_true_test, Ys_train, Ys_dev, Ys_true_test, _, _, _ = load_data(target, level = 2)

    ### get labels with correct target
    Ys_train = Ys_train[target]
    Ys_dev = Ys_dev[target]
    Ys_true_test = Ys_true_test[target]

    df_test = df_dev
    Ys_test = Ys_dev
#    small = list(
#        np.random.permutation(Ys_test.index.get_level_values('subject_id').values)
#    )[0:4]
#    df_test = df_test[df_test.index.get_level_values('subject_id').isin(small)]
#    Ys_test = Ys_test[Ys_test.index.get_level_values('subject_id').isin(small)]
#

    X_train = torch.from_numpy(to_3D_tensor(df_train).astype(np.float32))
    X_test = torch.from_numpy(to_3D_tensor(df_test).astype(np.float32))
    label_train = torch.from_numpy(Ys_train.values.astype(np.float32))
    label_test = torch.from_numpy(Ys_test.values.astype(np.float32))

    # take the mean over all patients over all 24 hours
    num_hours = 24
    x =  (df_train.loc[:, pd.IndexSlice[:, 'mean']] * np.where((df_train.loc[:, pd.IndexSlice[:, 'mask']] == 1).values, 1, np.NaN)).mean()
    X_mean = torch.Tensor( np.array([x.values] *num_hours )).unsqueeze(0)

    if use_gpu:
        X_train = dict(((gpu_id, X_train.to(gpu_id)) for gpu_id in device_ids))
        X_test = dict(((gpu_id, X_test.to(gpu_id)) for gpu_id in device_ids))
        X_mean = dict(((gpu_id, X_mean.to(gpu_id)) for gpu_id in device_ids))
        label_train = dict(((gpu_id, label_train.to(gpu_id)) for gpu_id in device_ids))
        label_test = dict(((gpu_id, label_test.to(gpu_id)) for gpu_id in device_ids))

    if use_gpu:
        device_type = get_gpu_id()
        X_train = X_train[device_type]
        label_train = label_train[device_type]    
        X_test = X_test[device_type]
        label_test = label_test[device_type]
        X_mean = X_mean[device_type]
    else:
        device_type = 'cpu'

    train_dataloader = create_dataloader(X_train, label_train, batch_size=h['batch_size'], shuffle = False)
    test_dataloader = create_dataloader(X_test, label_test, batch_size=h['batch_size'], shuffle = False)


    base_params = {'X_mean': X_mean, 'input_size': X_mean.shape[2], 
                   'device_id': device_type}
    base_params.update({"apply_sigmoid" : True})
    base_params.update({k: v for k, v in h.items() 
                        if k in ('cell_size', 'hidden_size', 'batch_size', 'dropout_p')})

def loss_fn(preds, targets):
    mse = torch.nn.MSELoss()
    return mse(torch.squeeze(preds), targets)

grud_wrapper = expm_grud_utils.ParallelGRUDWrapper(**base_params)

base_params.update({"use_bn" : False})
model = GRUD(**base_params)
if use_gpu:
    model.to(device_type)

model, losses = Train_Model(model, loss_fn, train_dataloader, 
    **{k: v for k, v in h.items() if k in (
        'num_epochs', 'patience', 'learning_rate', 'batch_size'
    )})



probabilities_dev, labels_dev = mmd_grud_utils.predict_proba(model, test_dataloader)
y_score = np.concatenate(probabilities_dev)
targets  = np.concatenate(labels_dev)

params = torch.cat([torch.flatten(p) for p in model.parameters()]) 
params2 = torch.stack([params]*3)


wrapper_probabilities_dev, wrapper_labels_dev = expm_grud_utils.predict_proba(grud_wrapper, params2, test_dataloader, 0)
wrapper_y_scores = np.concatenate(wrapper_probabilities_dev, axis = 1) # this is 3 x len(dev)

print("Wrapper scores match? (within 1e-6) ", all([(abs(y_score - wrapper_y_score)<1e-6).all() for wrapper_y_score in wrapper_y_scores]))
print("Wrapper scores identical? ", all([((y_score - wrapper_y_score)==0).all() for wrapper_y_score in wrapper_y_scores]))

print("Original AUC: ", roc_auc_score(targets, y_score))
print("Wrapper AUC: ", roc_auc_score(targets, wrapper_y_scores[0, :]))



