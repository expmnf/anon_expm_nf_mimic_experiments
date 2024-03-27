import torch.nn as nn
import torch, math

class LinearRegressionModel(torch.nn.Module):
    """
    A linear regression model (with no bias)
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1, bias = False)

    def forward(self, x):
        return self.linear(x)


# reference for basic logistic regression in Pytorch: https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
class LogisticRegression(torch.nn.Module): 
    """
    A basic logistic regression module.
    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class LogisticRegressionWrapper(torch.nn.Module): 
    """
    A basic logistic regression module.
    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionWrapper, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
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

    def forward(self, params, x):
        outputs = torch.sigmoid(self.call_linear_layer('linear', params, x))
        return outputs

    

   


