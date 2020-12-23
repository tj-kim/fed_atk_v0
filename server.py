# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:33:23 2020

@author: satya
"""

import matplotlib as plt
import numpy as np
import torch
import time

from copy import deepcopy


''' 
SERVER CLASS

Attributes:
  
- self.name             : Identification for server
- self.global_network   : Current global model stored in server
- self.update_vector    : Vector obtained by combining aggregated local parameters
- self.local_params     : Stored local parameters sent to server by clients


Methods:

- aggregate_params  : Aggregate local model parameters
    Input           : Local Model Parameter Repo
    Output          : Aggregated Parameters and metadata
    
- combine_params    : Combine local model parameters to make a compatible update vector for global model
    Input           : Aggregated Parameters and metadata
    Output          : Update Vector
        
- update_model      : Run update steps on global model
    Input           : Update Vector
    Output          : Updated Global Model
        
- distribute_model  : Post updated model with flag for clients to download
    Input           : Updated Global Model
    Output          : Metadata'''
    
class Server:
    
    def __init__(self,name,neck_network):
        
        self.name           = name
        self.neck           = neck_network
        self.update_dataset = dict()
        
        self.parameter_repo = dict()
        self.parameter_repo['download_time']    = 0
        self.parameter_repo['server']           = dict()
        self.parameter_repo['server']['model'] = neck_network
        
    def aggregate_params(self):
        
        self.parameter_repo['download_time'] = time.time()
        
        local_params    = []
        local_inputs    = []
        metadata        = []
        
        keys = list(self.parameter_repo.keys())
        keys.remove('download_time')
        keys.remove('server')
        # Get Local Parameters from updated Parameter Repo
        
        for key in keys:
            parameter = self.parameter_repo[key]['value']
            local_params.append(parameter)
            
            ip = self.parameter_repo[key]['input']
            local_inputs.append(ip)
            
            # Optional : Record Metadata
            metadata_key = 'time' # Change
            metadata.append(self.parameter_repo[key][metadata_key])
            
        ''' TO DO : ADHOC FUNCTIONS FOR AGGREGATION '''
        self.update_dataset['input'] = torch.stack(local_inputs)
        self.update_dataset['value'] = torch.Tensor(local_params)

        return self.update_dataset
        
    def combine_params(self):
        
        ''' TO DO : ADHOC FUNCTIONS FOR COMBINATION '''
        
        return self.update_vector
    
    def update_model(self,max_iterations):
        
        # Function to Update Model based on combined vector
        
        target_op = self.update_dataset['input'].clone()
        for i in range(max_iterations):
            
            op = self.neck.forward(self.update_dataset['input'])
            target = op.clone()+self.update_dataset['value']
            
            print(op)
            print(target)
            print()
            loss = self.neck.loss_critereon(op,target)
            print("Server Loss : ",loss.cpu().detach().numpy())
            
            self.neck.optimizer.zero_grad()
            loss.backward()
            self.neck.optimizer.step()
            
        updated_network = deepcopy(self.neck)
        return updated_network
        
    def distribute_model(self,updated_network):
        
        self.parameter_repo['server']['model']  = updated_network
        self.parameter_repo['server']['time']   = time.time()
        
        return self.parameter_repo
        