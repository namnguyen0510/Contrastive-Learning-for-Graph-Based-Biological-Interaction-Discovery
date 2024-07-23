import pandas as pd 
import numpy
import torch
import numpy as np
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
import os
import argparse


# PARAMETER
parser = argparse.ArgumentParser(description="Parse training parameters.")
parser.add_argument("--n_layers", type=int, default=4, help="Number of layers in the model.")
parser.add_argument("--loss_func", type=str, default='mse', help="Loss function to use.")
parser.add_argument("--optim_func", type=str, default='adamw', help="Optimizer function to use.")
parser.add_argument("--n_epochs", type=int, default=500, help="Number of epochs for training.")
parser.add_argument("--lr", type=float, default=0.3, help="Learning rate for the optimizer.")
args = parser.parse_args()

# SETTING PARAMETERS
n_layers = args.n_layers
loss_func = args.loss_func
optim_func = args.optim_func
n_epochs = args.n_epochs
lr = args.lr
exp_dirc = f'{n_layers}-{loss_func}-{optim_func}'
try:
    os.mkdir(exp_dirc)
except:
    pass

# LOAD BIOLOGICAL NETWORK
df = pd.read_csv('2_adjacency_matrix.csv')
print(df)
# LOAD MODEL
model = NN_CL_LD(df, n_layers).cuda()
# Print the parameters
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, Size: {param.size()}")
print(model())
br
# CONFIG. LOSS FUNCTION
if loss_func == 'mse':
    criterion = nn.MSELoss()
elif loss_func == 'mae':
    criterion = nn.L1Loss()
elif loss_func == 'kldiv':
    criterion = nn.KLDivLoss('mean')

# CONFIG. OPTIMIZER
if optim_func == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif optim_func == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optim_func == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=lr)


best_loss = 10e6
running_loss = []
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model()
    loss = criterion(outputs[0], outputs[1])
    loss.backward()
    optimizer.step()
    print('EPOCH: {}|CL Loss: {}'.format(epoch, loss.item()))
    if loss < best_loss:
        torch.save(model.state_dict(), f'{exp_dirc}/1_best_model.pth')
        best_loss = loss
    running_loss.append(loss.item())

exp_hist = pd.DataFrame(np.array(running_loss))
exp_hist.to_csv(f'{exp_dirc}/exp_hist.csv')
# Load the saved model state dict
model.load_state_dict(torch.load(f'{exp_dirc}/1_best_model.pth'))
# Set the model in evaluation mode (important if using layers like dropout or batchnorm)
model.eval()
syn_graph = model.inference()
# OUTPUT SYNTHESIZED GRAPH
outdf = pd.DataFrame(syn_graph.detach().cpu().numpy())
outdf.columns = df.columns
outdf.index = df.columns
outdf.to_csv(f'{exp_dirc}/0_result_synthesized_graph.csv', index = False)