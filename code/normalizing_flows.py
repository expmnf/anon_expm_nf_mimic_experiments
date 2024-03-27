import torch
import torch.nn as nn
from torch.autograd import Variable


# Function to sample from base distribution
def random_normal_samples(n, dim=1, std = 1, device = 'cpu'):
    return torch.zeros(n, dim, device = device).normal_(mean=0, std=std)

# def random_uniform_samples(n, dim = 1, u = 1): 
# 	return torch.zeros(n, dim).uniform_(-u, u)

class PlanarFlow(nn.Module): # base class for a single Planar Flow layer

    """
    A single planar flow, 
    - computes `T(z) = z + h( <z,w> + b)u` where parameters `w,u` are vectors, `b` is a scalar, `h` is tanh activation function. 
    - log(det(|jacobian_T|)))
    """
    def __init__(self, D):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.h = torch.tanh
        self.dim = D
        self.init_params()

    def init_params(self):
        self.w.data.uniform_(-0.01, 0.01)
        self.b.data.uniform_(-0.01, 0.01)
        self.u.data.uniform_(-0.01, 0.01)

    def forward(self, z):
        linear_term = torch.mm(z, self.w.T) + self.b
        return z + self.u * self.h(linear_term)

    def h_prime(self, x):
        """
        Derivative of tanh
        """
        return 1 - (self.h(x)).pow(2)

    def psi(self, z):
        # auxiliary vector computed for the log_det_jacobian
        # f'(z) = I + u \psi^t, so det|f'| = 1 + u^t \psi
        inner = torch.mm(z, self.w.T) + self.b
        return self.h_prime(inner) * self.w

    def log_det_jac(self, z):
        inner = 1 + torch.mm(self.psi(z), self.u.T)
        return torch.log(torch.abs(inner))

    ## Could implement an inverse method. 
    ## could implement a p_x(x) = p_u(f^{-1}(x)) |det f'(u)|


class NormalizingFlow(nn.Module):
    """
    A normalizng flow composed of a sequence of planar flows.
    superclass is torch.nn
    """
    def __init__(self, D, n_flows=2):
        """Initiates a NF class with a sequence of planar flows. 
            - runs the super class .init
            - creates a new attribute, `self.flows` that is ModuleList (a PyTorch object for having lists of PyTorch objects) of the planar flows in each layer.
        Args:
            D (int): dimension of this flow
            n_flows (int, optional): How many layers of the NF, defaults to 2.
        """
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList(
            [PlanarFlow(D) for _ in range(n_flows)])
        self.dim = D

    def sample(self, base_samples):
        """
        Transform samples from a simple base distribution
        by passing them through a sequence of Planar flows.

        Args:
            base_samples (torch.tensor): samples from base distribution 

        Returns:
            samples (torch.tensor): transformed samples 
        """
        samples = base_samples
        for flow in self.flows:
            samples = flow(samples)
        return samples

    def forward(self, x):
        """
        Computes and returns: 
        -  T(x) = f_k\circ ... \circ f_1(x) (the transformed samples)
        - \log( |\det T'(x)|) = sum \log[ \det |f_i'(x_i)|]
        """
        sum_log_det = 0
        transformed_sample = x

        for i in range(len(self.flows)):
            log_det_i = (self.flows[i].log_det_jac(transformed_sample))
            sum_log_det += log_det_i
            transformed_sample = self.flows[i](transformed_sample)

        return transformed_sample, sum_log_det
    
    ## Could implement an inverse method. 
    ## could implement a p_x(x) = p_u(f^{-1}(x)) |det f'(u)|


class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    - z_size (int): the dimension  of the flow 
    - m (int): the rank of $Q$, with `m <= z_size` defaulting to `m = z_size`. 
    - hh (int): the Householder number (how many HH reflections to use in creating Q. It will default to `z_size-1`)
    - use_tanh (bool): if `True`,  `h=tanh`; else `h=relu` (TODO, add this)
    """
    def __init__(self, z_size, m=None, hh=None, use_tanh = True):

        super(Sylvester, self).__init__()

        ## instantiate attributes: 
        self.z_size = z_size
        
        if not m: self.m = z_size
        else: self.m = m
        
        if not hh: self.hh = z_size - 1
        else: self.hh = hh
        
        if use_tanh: self.h = nn.Tanh()
        else: print("NEED TO ADD RELU OPTION STILL")
        
        self.use_tanh = use_tanh

        ## instantiate parameters 
        ## z--> z + Ah(Bz + c), A = Q R1, B = R2 Q^t, Q = prod (1-2 v_i v_i^T)
        self.v = nn.Parameter(torch.Tensor(self.hh, self.z_size), requires_grad = True) # each row is a v_i vector used to make Q    
        self.Rs = nn.Parameter(torch.Tensor(self.m, self.m), requires_grad = True) # upper half including diagonal is R1, bottom half not including diag is bottom half of R2. 
        self.r2diag = nn.Parameter(torch.Tensor(self.m), requires_grad = True) # diagonal for R2.         
        self.c = nn.Parameter(torch.Tensor(1, self.m), requires_grad = True) # vector inside h
        
        self.init_params()
        
        ## make masks needed for creating R1, R2, and register the masks so they ship w/ the model. 
        triu_mask_1 = torch.triu(torch.ones(self.m, self.m), diagonal=0)
        triu_mask_2 = torch.triu(torch.ones(self.m, self.m), diagonal=1)
        self.register_buffer('triu_mask_1', Variable(triu_mask_1))
        self.triu_mask_1.requires_grad = False
        self.register_buffer('triu_mask_2', Variable(triu_mask_2))
        self.triu_mask_2.requires_grad = False
        diag_idx = torch.arange(0, self.m).long()        
        self.register_buffer('diag_idx', diag_idx)
        self.diag_idx.requires_grad = False

        _eye = torch.eye(z_size, z_size) ## buffer the identity so it's easy to grab. 
        self.register_buffer('_eye', Variable(_eye))
        self._eye.requires_grad = False 
        
    def init_params(self): # randomly initialize the values. 
        self.v.data.uniform_(-1, 1)
        self.Rs.data.uniform_(-1, 1)
        self.r2diag.data.uniform_(-1,1)
        self.c.data.uniform_(-1, 1)    
    
    def der_h(self, x):
        if self.use_tanh: 
            return self.der_tanh(x)
        else: print("NEED TO ADD RELU OPTION STILL")

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def create_rs(self):
        # returns r1, r2, both are (self.m,self.m) upper triangular, from self.R2.data, self.r2diag parameters. 
        r1 = self.triu_mask_1 * self.Rs
        r2 = self.triu_mask_2 * self.Rs
        r2[self.diag_idx, self.diag_idx] = self.r2diag
        return r1, r2

    def create_q(self): 
        # returns q (z_shape, m ), ortho.n. matrix prod_i (I - 2v_i v_i^t)
        v = self.v # shape (hh, z_size)
        norms = torch.norm(v, p = 2, dim = 1, keepdim = True)
        v = torch.div(v, norms) # now every row has norm 1. 
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1)) # (hh, z_size , z_size) 
        ivvT = self._eye - 2*vvT # (hh, z_size , z_size) 
        q = ivvT[0] # (z_size, z_size)
        for i in range(1, ivvT.shape[0]):# for all the other vvT matrices 
            q = torch.mm(q, ivvT[i]) # multiply them
        q = q[ : , : self.m] # (z_size , m)
        return q  # behold our beautiful orthonormal matrix

    def forward(self, z, sum_ldj=True):
        """
        z (torch.Tensor): shape is (batch_size, z_size). 
        
        Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. 
        This computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param zk: shape: (batch_size, z_size)
        returns: f(z), log_det_j
        notes:         
        - r1: shape (m, m)
        - r2: shape: (m, m)
        - q_ortho: shape (z_size , m)
        - self.c: shape: (1, self.z_size)
        """
        assert z.shape[1] == self.z_size
        
        ## get matrices R1, R2, Q, A, B
        r1, r2 = self.create_rs()
        q = self.create_q()
        Bt = torch.mm(q, r2.T) # B.T
        A = torch.mm(q, r1) # (z_size, m)
        Btzc = torch.mm(z, Bt) + self.c # (batch_size, m) 
        z = torch.mm(self.h(Btzc), A.T) + z # this is f(z), shape = (batch_size, z_size)

        ## now compute log_det_J  
        rrii = r1[self.diag_idx, self.diag_idx]*r2[self.diag_idx, self.diag_idx]
        ldj = self.der_h(Btzc) * rrii + 1
        ldj = ldj.abs().log().sum(1) # shape (z_size)

        return z, ldj


class NormalizingFlowSylvester(nn.Module):
    """
    A normalizng flow composed of a sequence of Sylvester flows.
    superclass is torch.nn
    """
    def __init__(self, z_size, m = None, hh=None,  n_flows=2, use_tanh = True):
        """
        Args:
            z_size (int): dimension of this flow
            n_flows (int, optional): How many layers of the NF, defaults to 2.
        """
        super(NormalizingFlowSylvester, self).__init__()
        
        ## instantiate attributes: 
        self.z_size = z_size
        
        if not m: self.m = z_size
        else: self.m = m
        
        if not hh: self.hh = z_size - 1
        else: self.hh = hh
        
        if use_tanh: self.h = nn.Tanh()
        else: print("NEED TO ADD RELU OPTION STILL")
        
        self.use_tanh = use_tanh
        
        self.flows = nn.ModuleList(
            [Sylvester(self.z_size, m = self.m, hh = self.hh, use_tanh = self.use_tanh) for _ in range(n_flows)])
        

    def forward(self, z):
        """
        Computes and returns: 
        -  T(x) = f_k\circ ... \circ f_1(x) (the transformed samples)
        - \log( |\det T'(x)|) = sum \log[ \det |f_i'(x_i)|]
        """
        ldj = 0

        for i in range(len(self.flows)):
            z, ldj_i = self.flows[i].forward(z)
            ldj += ldj_i
            
        return z, ldj
