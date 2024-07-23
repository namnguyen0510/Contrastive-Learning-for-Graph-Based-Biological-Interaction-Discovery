import pandas as pd 
import numpy
import torch
import numpy as np
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class NN_CL_LD(nn.Module):
    def __init__(self, data, n_layers):
        super(NN_CL_LD, self).__init__()
        # DATALOADER
        n = len(data)
        self.data = data
        # MODEL
        self.layers = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        for i in range(n_layers):
            self.layers["layer_weight_{}".format(i)] = nn.Parameter(torch.rand(n,n, requires_grad = True))
            self.biases["layer_bias_{}".format(i)] = nn.Parameter(torch.rand(n,n, requires_grad = True))

    def forward(self):
        _, self.x_1 = permute_df(self.data)
        _, self.x_2 = permute_df(self.data)
        self.x_1 = self.x_1.double()
        self.x_2 = self.x_2.double()
        for i in range(len(self.layers)):
            self.x_1 = self.layers["layer_weight_{}".format(i)]*self.x_1 + self.biases["layer_bias_{}".format(i)]
            self.x_2 = self.layers["layer_weight_{}".format(i)]*self.x_2 + self.biases["layer_bias_{}".format(i)]
        self.x_1, self.x_2 = get_symmetric(self.x_1), get_symmetric(self.x_2)
        self.x_1 = self.x_1 - torch.diag_embed(torch.diag(self.x_1))
        self.x_2 = self.x_2 - torch.diag_embed(torch.diag(self.x_2))
        return self.x_1, self.x_2

    def inference(self):
        x = torch.tensor(self.data.to_numpy()).double()
        for i in range(len(self.layers)):
            x = self.layers["layer_weight_{}".format(i)]*x + self.biases["layer_bias_{}".format(i)]
        x = torch.softmax(x,dim=0)
        x =  get_symmetric(x)
        x = x - torch.diag_embed(torch.diag(x))
        return x