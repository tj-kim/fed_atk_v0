import torch
import torch.nn as nn
import torch.optim as optim

from config import config
import numpy as np

class Head_Network(nn.Module):
    
    def __init__(self,architecture,load_wt=True):
        super(Head_Network, self).__init__()
        
        self.architecture       = architecture
        self.load_weights       = load_wt
        self.network = nn.ModuleList([])
    
        for idx in range(len(self.architecture)-1):
            self.network.append(nn.Linear(self.architecture[idx], self.architecture[idx+1]))
            
            if(self.load_weights):
                weight_file_name = 'weights/layer_'+str(idx)+'_weights.npy'
                layer_weights = nn.Parameter(torch.Tensor(np.load(weight_file_name)))
                self.network[idx].weight = layer_weights
                
                bias_file_name = 'weights/layer_'+str(idx)+'_bias.npy'
                bias_weights = nn.Parameter(torch.Tensor(np.load(bias_file_name)))
                self.network[idx].bias = bias_weights
                
            else:
                torch.nn.init.xavier_uniform_(self.network[idx].weight)
        
        self.network        = self.network.cuda()
        self.activation     = nn.LeakyReLU(-0.1)
        self.softmax        = nn.LogSoftmax(dim=1)
        self.loss_critereon = nn.NLLLoss()
        self.optimizer      = optim.Adam(self.parameters(),lr=config['head_lr'])
            
    def forward(self,x):
        
        x = x.cuda()
        for layer in self.network:
            x = layer(x)
            x = self.activation(x)
            
        x = self.softmax(x)
        
        return x