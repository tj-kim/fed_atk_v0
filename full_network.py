import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Full_Network(nn.Module):
    
    def __init__(self,architecture):
        super(Full_Network, self).__init__()
        
        self.architecture      = architecture
        
        self.network = nn.ModuleList([])
    
        for idx in range(len(self.architecture)-1):
            self.network.append(nn.Linear(self.architecture[idx], self.architecture[idx+1]))
            torch.nn.init.xavier_uniform_(self.network[idx].weight)
          
        self.network = self.network.cuda()
        self.activation     = nn.LeakyReLU(-0.1)
        self.softmax        = nn.LogSoftmax(dim=1)
        self.loss_critereon = nn.NLLLoss()
        self.optimizer      = optim.SGD(self.parameters(),lr=1e-5)
            
    def forward(self,x):
        x = x.cuda()
        for layer in self.network:
            x = layer(x)
            x = self.activation(x)
            
        x = self.softmax(x)
        
        return x