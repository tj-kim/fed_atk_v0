# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:33:23 2020

@author: satya
"""

import matplotlib as plt
import numpy as np
import torch
import time

from torch.autograd import Variable
from copy import deepcopy

from config import config

''' 
CLIENT CLASS

Attributes:
  
- self.name         : Identification for the client
- self.network      : Local model that is used by a client to infer from
- self.dataset      : Locally available data drawn from a complete dataset
- self.params       : Structure to contain information to be uploaded to server


Methods:

- download_params   : Download parameters from the server and update local model
    Input           : None
    Output          : Global Model
    
- forward_pass      : Infer the output using the local model and calculate loss
    Input           : Local Model
    Output          : Model Loss & related metadata
    
- backward_pass     : Run backpropagation and update the local model
    Input           : Model Loss & related metadata
    Output          : Updated Local Model

- upload_params     : Upload model parameters to the server
    Input           : Updated Local Model
    Output          : Success Flag

'''

class Client:
    
    def __init__(self,name,neck,head):
        
        self.name = name
        self.neck = neck
        self.head = head
        
        self.ip = None
        self.neck_op = None
        self.head_ip = None
        self.op = None
        
        self.parameter = int(0)
        self.param_repo = dict()
        
        self.iteration_count = 1
        self.download_time = 0
        
    def detach(self,input_tensor):
        return input_tensor.cpu().detach().numpy()
        
    def set_global_repo(self,global_repo):
        self.global_repo = global_repo
        
    def load_dataset(self,loader):
        
        self.loader = loader
        
    def download_params(self):
        
        # Function to Check Parameter Repo for Updates
        self.neck = self.global_repo['server']['model']
        
        return 0
        
    def forward_pass(self):
        
        self.loader.empty_repo()
        image_data = self.loader.load_images(batch_size=config['batch_size'])
        
        self.ip          = torch.Tensor(image_data['input'])
        self.target      = torch.Tensor(image_data['label']).type(torch.LongTensor).cuda()
    
        self.neck_op = self.neck.forward(self.ip)
        
        self.head_ip = self.neck_op.clone()
        self.head_ip = Variable(self.head_ip,requires_grad=True)
        
        self.op      = self.head.forward(self.head_ip)
        
    def backward_pass(self):
        
        loss = self.head.loss_critereon(self.op,self.target)
        print(" Loss : ",loss.cpu().detach().numpy())
        
        self.head.optimizer.zero_grad()
        loss.backward()
        self.head.optimizer.step()
        
        # Calculate Running Average of Gradients
        if(type(self.parameter)==int):
            self.parameter = np.mean(self.detach(self.head_ip.grad),0)
        else:
            self.parameter = np.mean(self.detach(self.head_ip.grad),0)
            #self.parameter = (self.parameter*(self.iteration_count-1)+np.mean(self.detach(self.head_ip.grad),0))/self.iteration_count
        
        print('Client : ',self.parameter)
        
    def upload_params(self):
        
        # Save Gradient in an internal repository
        self.param_repo['input']    = self.ip
        self.param_repo['value']    = self.parameter
        self.param_repo['time']     = int(time.time())
        
        self.iteration_count+=1
        
        # Function to uplaod stored parameters
        self.global_repo[self.name] = deepcopy(self.param_repo)
        
        # Reset Parameters if they have already been used by the server
        if(not(self.global_repo['download_time']==self.download_time)):
            print('Reset')
            self.param_repo         = dict()
            self.parameter          = int(0)
            self.iteration_count    = 1
            self.download_time      = self.global_repo['download_time']
        
        return 0