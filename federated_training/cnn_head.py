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


class CNN_Head(nn.Module):
    
    def __init__(self,mode):
        super(CNN_Head, self).__init__()
        
        with open(r'configs/config.yaml') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        
        
        self.mode = mode
        self.architecture      = []
        
        self.network = nn.ModuleList([])
        
        self.activation     = nn.LeakyReLU(float(self.config['leaky_alpha']))
        self.softmax        = nn.LogSoftmax(dim=1)
        self.loss_critereon = nn.NLLLoss()
        self.dropout        = nn.Dropout(p=self.config['dropout'])
        
        for layer in self.config['head_architecture']:
            self.network.append(Linear(layer[0],layer[1]))
        
        self.optimizer      = optim.Adam(self.parameters(),lr=float(self.config['head_lr']))
        
        for idx in range(len(self.architecture)-1):    
            torch.nn.init.xavier_uniform_(self.network[idx].weight)
		
        if(self.mode=='cuda'):
            self.network 		= self.network.cuda()
        else:
            self.network 		= self.network	
            
        self.gradients = None
        self.hook_mode = 'train'
        
        self.pretrain = False
        
    def activations_hook(self, grad):
        self.gradients = grad
         
    def forward(self,x):
        
        if(self.mode=='cuda'):
            x = x.cuda()
        
        if(self.hook_mode=='train' and not(self.config['mode']=='pingpong')):
            x.register_hook(self.activations_hook)
        
        # TJ Edit Jan 7 2021 - Iterative forward pass for different architectures
        for i in range(len(self.config['head_architecture'])):
            x = self.network[i](x)
            x = self.activation(x)
            if(self.pretrain==True):
                x = self.dropout(x)
        
        x = self.softmax(x)
        
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