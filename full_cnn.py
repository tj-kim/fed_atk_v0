import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.optim as optim

class Full_CNN(nn.Module):
    
    def __init__(self,mode):
        super(Full_CNN, self).__init__()
        
        self.mode = mode
        self.architecture      = []
        
        self.network = nn.ModuleList([])
        
        self.activation     = nn.LeakyReLU(-0.1)
        self.softmax        = nn.LogSoftmax(dim=1)
        self.loss_critereon = nn.NLLLoss()
        self.pooling = MaxPool2d(kernel_size=3, stride=3)
        
        self.network.append(Conv2d(1,  64, kernel_size=5, stride=1, padding=1))
        self.network.append(self.pooling)
        self.network.append(self.activation)
        self.network.append(Conv2d(64, 64, kernel_size=5, stride=1, padding=1))
        self.network.append(self.pooling)
        self.network.append(self.activation)
        
        self.network.append(Linear(256,256))
        self.network.append(self.activation)
        self.network.append(Linear(256,63))
        self.network.append(self.activation)
        
        self.optimizer      = optim.Adam(self.parameters(),lr=2e-4)
        
        for idx in range(len(self.architecture)-1):    
            torch.nn.init.xavier_uniform_(self.network[idx].weight)
		
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
        x = self.network[1](x)
        x = self.network[2](x)
        x = self.network[3](x)
        x = self.network[4](x)
        x = self.network[5](x)
        x = x.view(x.size(0), -1)
        x = self.network[6](x)
        x = self.network[7](x)
        x = self.network[8](x)
        x = self.network[9](x)
        
        x = self.softmax(x)
        
        return x

    def get_activations_gradient(self):
    
        return self.gradients