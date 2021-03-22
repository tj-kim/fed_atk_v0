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
    
class CNN_Neck(nn.Module):
    
    def __init__(self,mode):
        super(CNN_Neck, self).__init__()
        
        with open(r'configs/config.yaml') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        
        self.mode = mode        
        self.network = nn.ModuleList([])
        
        if(self.config['mode']=='pingpong'):
            self.loss_critereon = nn.NLLLoss()
        else:
            self.loss_critereon = nn.L1Loss(size_average=False)
        self.activation     = nn.LeakyReLU(float(self.config['leaky_alpha']))
        self.pooling = MaxPool2d(kernel_size=self.config['maxpool_kernel_size'], stride=self.config['maxpool_stride'])
        
        # Add convolutional Layers
        init_idx = 0
        for layer in self.config['neck_architecture']:
            self.network.append(Conv2d(layer[0], layer[1], kernel_size=self.config['kernel_size'], 
                                       stride=self.config['stride'], padding=self.config['padding']))
        
        if(self.mode=='cuda'):
            self.network.append(nn.BatchNorm1d(num_features=self.config['neck_architecture'][-1][-1]*25).cuda())
        else:
            self.network.append(nn.BatchNorm1d(num_features=self.config['neck_architecture'][-1][-1]*25))
        
        # Add Linear Layers
        for layer in self.config['neck_architecture_lin']:
            self.network.append(Linear(layer[0],layer[1]))
        
        self.optimizer      = optim.Adam(self.parameters(),lr=float(self.config['head_lr']))
        
        # Initialize weights for layers that don't raise error
        for idx in range(len(self.network)):  
            try:
                torch.nn.init.xavier_uniform_(self.network[idx].weight)
            except:
                continue
		
        
        if(self.mode=='cuda'):
            self.network 		= self.network.cuda()
        else:
            self.network 		= self.network	
            
        self.gradients = None
        
        # Bring in head components
        self.hook_mode = 'train'
        self.pretrain = False
        
    def activations_hook(self, grad):
        self.gradients = grad
         
    def forward(self,x):
                  
        if(self.mode=='cuda'):
            x = x.cuda()
        
        # TJ Edit Jan 7 2021 - Iterative forward pass for different architectures
        for i in range(len(self.config['neck_architecture'])):
            x = self.network[i](x)
            x = self.pooling(x)
            x = self.activation(x)
           
        x = x.view(x.size(0), -1)
        x = self.network[i+1](x)
        
        # Head Mode
        if(self.hook_mode=='train' and not(self.config['mode']=='pingpong')):
            x.register_hook(self.activations_hook)
        
        for j in range(len(self.config['neck_architecture_lin'])):
            x = self.network[j+i+2](x)
            x = self.activation(x)
            if(self.pretrain==True):
                x = self.dropout(x)
        
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
            
            
        
            
            