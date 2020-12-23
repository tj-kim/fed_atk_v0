# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:11:48 2020

@author: satya
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.optim as optim
import yaml

with open(r'config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
class CNN_Neck(nn.Module):
    
    def __init__(self,mode):
        super(CNN_Neck, self).__init__()
        
        self.mode = mode        
        self.network = nn.ModuleList([])
        
        if(config['mode']=='pingpong'):
            self.loss_critereon = nn.NLLLoss()
        else:
            self.loss_critereon = nn.L1Loss(size_average=False)
        self.activation     = nn.LeakyReLU(float(config['leaky_alpha']))
        self.pooling = MaxPool2d(kernel_size=config['maxpool_kernel_size'], stride=config['maxpool_stride'])
        for layer in config['neck_architecture']:
            self.network.append(Conv2d(layer[0], layer[1], kernel_size=config['kernel_size'], stride=config['stride'], padding=config['padding']))
        
        self.optimizer      = optim.Adam(self.parameters(),lr=float(config['head_lr']))
        
        for idx in range(len(self.network)):    
            torch.nn.init.xavier_uniform_(self.network[idx].weight)
		
        self.network.append(nn.BatchNorm1d(num_features=config['neck_architecture'][-1][-1]*25).cuda())
        
        if(self.mode=='cuda'):
            self.network 		= self.network.cuda()
        else:
            self.network 		= self.network	
            
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad
         
    def forward(self,x):
        x = x.cuda()
        
        x = self.network[0](x)
        x = self.pooling(x)
        x = self.activation(x)
        x = self.network[1](x)
        x = self.pooling(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.network[2](x)
        
        return x

    def get_activations_gradient(self):
        return self.gradients
    
    def get_weights(self):
        
        weight_log = []
        
        for layer in self.network:
            
            weights = layer.weight.cpu().detach().numpy()
            bias    = layer.bias.cpu().detach().numpy()
            weight_log.append([weights,bias])
        
        return weight_log
    
    def load_weights(self,weights):
        
        with torch.no_grad():
            for idx in range(len(self.network)):        
                self.network[idx].weight.copy_(torch.from_numpy(weights[idx][0]))
                self.network[idx].bias.copy_(torch.from_numpy(weights[idx][1]))
            
            
        
            
            