# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 21:25:35 2020

@author: satya
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

from config import config
import numpy as np

class Neck_Network(nn.Module):
    
    def __init__(self,architecture,load_wt=True):
        super(Neck_Network, self).__init__()
        
        self.architecture      = architecture
        self.load_weights       = load_wt
        self.network = nn.ModuleList([])
    
        for idx in range(len(self.architecture)-1):
            self.network.append(nn.Linear(self.architecture[idx], self.architecture[idx+1]))
            
            if(self.load_weights):
                weight_file_name = 'weights/neck_layer_'+str(idx)+'_weights.npy'
                layer_weights = nn.Parameter(torch.Tensor(np.load(weight_file_name)))
                self.network[idx].weight = layer_weights
                
                bias_file_name = 'weights/neck_layer_'+str(idx)+'_bias.npy'
                bias_weights = nn.Parameter(torch.Tensor(np.load(bias_file_name)))
                self.network[idx].bias = bias_weights
            else:
                torch.nn.init.xavier_uniform_(self.network[idx].weight)
            
        self.activation     = nn.LeakyReLU(-0.1)
        self.loss_critereon = nn.MSELoss(reduction='sum')
        self.optimizer      = optim.Adam(self.parameters(),lr=config['neck_lr'])
            
    def forward(self,x):
        
        for layer in self.network:
            x = layer(x)
            x = self.activation(x)
                        
        return x