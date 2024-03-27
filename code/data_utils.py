import pandas as pd 
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import codecs, datetime, json, os, pickle, copy
import numpy as np

class DatasetMaker(Dataset): 
    def __init__(self, X, y): 
        """PyTorch provides a Dataset class for housing data. 
        This class instantiates the torch Dataset class.
        Args:
            X (numpy or torch array): feature design matrix
            y (numpy or torch array): targets/labels vector
        """
        super(DatasetMaker, self).__init__()
        if isinstance(X, type(torch.tensor([]))):
            self.X = X
        elif isinstance(X, type(pd.DataFrame([]))):
            self.X = torch.tensor(X.to_numpy(), dtype = torch.float32)
        else:     
            self.X = torch.tensor(X, dtype = torch.float32)
        if isinstance(y, type(torch.tensor([]))):
            self.y = y
        elif isinstance(y, type(pd.DataFrame([]))) or isinstance(y, type(pd.Series([]))):
            self.y = torch.tensor(y.to_numpy(), dtype = torch.float32) 
        else: 
            self.y = torch.tensor(y, dtype = torch.float32)

    def __len__(self): 
        return self.y.shape[0]

    def __getitem__(self, idx): 
        x = self.X[idx]
        y = self.y[idx]
        return x, y

    def to(self, gpu_id):
        self.X = self.X.to(gpu_id)
        self.y = self.y.to(gpu_id)
        return copy.copy(self)


# DatasetMaker class allows us to use PyTorches DataLoader class which 
# shuffles data, organizes data into batches, and parallelizes this process for us. 

# Example use: 
# csvpath = Path('.',  'data', 'height-weight.csv' )
# df = pd.read_csv(csvpath)
# X = df.iloc[:,0].values
# y = df.iloc[:,1].values 
# data = DatasetMaker(X_train, y_train)
# dataloader = DataLoader(dataset = data, batch_size = 50, shuffle = True)



# functions for saving/opening objects
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def jsonify(obj, out_file):
    """
    Inputs:
    - obj: the object to be jsonified
    - out_file: the file path where obj will be saved
    This function saves obj to the path out_file as a json file.
    """
    json.dump(obj, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4,
              cls = NpEncoder)


def unjsonify(in_file):
    """
    Input:
    -in_file: the file path where the object you want to read in is stored
    Output:
    -obj: the object you want to read in
    """
    obj_text = codecs.open(in_file, 'r', encoding='utf-8').read()
    obj = json.loads(obj_text)
    return obj


def keystoint(x):
    try:
        return {int(k): v for k, v in x.items()}
    except:
        return x
    
        
def unjsonify_int_keys(in_file):
    obj_text = codecs.open(in_file, 'r', encoding='utf-8').read()
    obj = json.loads(obj_text, object_hook=keystoint)
    return obj


def picklify(obj, filepath):
    """
    Inputs:
    - obj: the object to be pickled
    - filepath: the file path where obj will be saved
    This function pickles obj to the path filepath.
    """
    pickle_file = open(filepath, "wb")
    pickle.dump(obj, pickle_file)
    pickle_file.close()
    # print "picklify done"


def unpickle(filepath):
    """
    Input:
    -filepath: the file path where the pickled object you want to read in is stored
    Output:
    -obj: the object you want to read in
    """
    pickle_file = open(filepath, 'rb')
    obj = pickle.load(pickle_file)
    pickle_file.close()
    return obj

