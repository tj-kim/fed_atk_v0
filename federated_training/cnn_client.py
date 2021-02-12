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

from federated_training import config
from federated_training.utilities import freeze_layers
import threading
import yaml
import torch.nn as nn

import copy

with open(r'configs/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

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

class Client():
    
    def __init__(self,name,neck,head,update_schedule,model_queue,param_queue,loader,data_manager=None):
        
        self.name = name
        self.head = head
        self.neck = neck
        
        self.data_repo =  {'name'   :self.name,
                           'inputs' :None,
                           'params' :None,
                           'network':None,
                           'time'   :0}
        
        self.model_data = None
        
        self.ip = None
        self.neck_op = None
        self.head_ip = None
        self.op = None
        
        self.internal_repo = {'inputs':[],'params':[]}
        self.update_count = -1
        self.download_time = 0
        
        self.batch_size = config['client_batch_size']
        self.loader = loader
        self.losses = []
        self.old_time = 0
        self.new_time = 0
        self.cumulative_gradient = np.zeros(1600)
        self.update_schedule = update_schedule
        self.iteration_count = 0
        
        self.model_queue = model_queue
        self.param_queue = param_queue
        
        self.new_count=-1
        self.training_accuracy = 0.0
        self.training_status = False
        self.data_manager = data_manager
        self.train_log = []
        self.test_log = []
        self.complete = False
        
        self.ip_to_upload       = []
        self.param_to_upload    = []
        
    def detach(self,input_tensor):
        return input_tensor.cpu().detach().numpy()
        
    def import_loader(self,loader):
        self.loader = loader
        
    def check_lock(self):
        pass    
        return 0
        
    def get_network(self):
    
        with torch.no_grad():
            weight_data = []
            bias_data   = []
            for idx in range(len(config['head_architecture'])):
                weight_data.append(self.head.network[idx].weight)
                bias_data.append(self.head.network[idx].bias)
                
        return [weight_data,bias_data]
        
        
    def download_params(self):
        ''' Check Queue for new mdoels and update local model '''
        
        # Keep Checking until updated model is found
        while(self.update_count==self.new_count and self.training_status==False): 
            self.model_data = copy.deepcopy(self.model_queue.get())
            self.new_count  = self.model_data['update_count']
            self.training_status = self.model_data['complete']
            self.model_queue.put(copy.deepcopy(self.model_data)) 
        
        # Update local model
        self.update_count = self.model_data['update_count']
        self.clients_to_upload = self.update_schedule[self.update_count]
        self.complete   = self.model_data['complete']
        
        if(self.complete): return 1
        
        if(self.update_count<config['pretrain']):
            pass
        else:
            with torch.no_grad():
                
                # Added the length of neck architecture as well
                if(config['get_neck']):
                    for idx in range(len(config['neck_architecture'])):
                        self.neck.network[idx].weight.copy_(torch.from_numpy(self.model_data['neck_model'][idx][0]))
                        self.neck.network[idx].bias.copy_(torch.from_numpy(self.model_data['neck_model'][idx][1]))
                        
                    for idx2 in range(len(config['neck_architecture_lin'])):
                        new_idx = idx2 + idx + 2
                        self.neck.network[new_idx].weight.copy_(torch.from_numpy(self.model_data['neck_model'][new_idx][0]))
                        self.neck.network[new_idx].bias.copy_(torch.from_numpy(self.model_data['neck_model'][new_idx][1]))
                        
                        
                if(config['get_head']):
                    for idx in range(len(config['head_architecture'])):
                        self.head.network[idx].weight.data.copy_(torch.tensor(self.model_data['head_model'][idx][0]))
                        self.head.network[idx].bias.data.copy_(torch.tensor(self.model_data['head_model'][idx][1]))
        return 0
        
    def forward_pass(self):
        ''' Upload data in Phase I and Train Network in Phase II'''
        
        if(self.complete): return 1
        
        for i in range(config['client_iterations']):
            image_data = self.loader.load_batch(batch_size=self.batch_size)
            self.ip          = torch.Tensor(image_data['input']).reshape(self.batch_size,1,28,28)
            
            if(torch.cuda.is_available()):
                self.target      = torch.Tensor(image_data['label']).type(torch.LongTensor).cuda()
            else:
                self.target      = torch.Tensor(image_data['label']).type(torch.LongTensor)
            

            self.neck_op = self.neck.forward(self.ip)
            
            if(config['mode']=='gradient'):
                self.neck_op = self.neck_op.clone()
                self.neck_op = Variable(self.neck_op,requires_grad=True)
                
            self.op      = self.head.forward(self.neck_op)
            
            loss = self.head.loss_critereon(self.op,self.target)
            
            self.head.optimizer.zero_grad()
            
            if(config['client_train']):
                self.neck.optimizer.zero_grad()
                
            loss.backward()
            self.head.optimizer.step()
            
            if(config['client_train']):
                print('Ja')
                self.neck.optimizer.step()
                
            # Save Data to Upload
            for i in range(self.batch_size):
                self.ip_to_upload+=list(self.detach(self.ip[i]))
                #self.param_to_upload.append(list(np.mean(np.array(self.detach(self.neck_op.grad)),0)))
                self.param_to_upload+=list(np.array(self.detach(self.target)))
            
            
            self.total_loss += loss.cpu().detach().numpy()
        
    def upload_params(self):
        
        if(self.complete): return 1
        
        ''' Upload to Queue '''
        
        if(self.name in self.clients_to_upload):
            if(np.random.random()<config['upload_prob']):
                idx_to_share = np.random.randint(0,self.batch_size,self.batch_size)
                if(self.param_queue.qsize()<config['num_clients']):
                
                    self.data_repo['inputs'] = np.array([self.ip_to_upload[idx] for idx in idx_to_share])
                    self.data_repo['params'] = np.array([self.param_to_upload[idx] for idx in idx_to_share])
                    self.data_repo['time']   = int(time.time())
                    self.data_repo['network']= self.get_network()
                    self.data_repo['ids']     = [self.name]*len(self.param_to_upload)
                    
                    to_send = copy.deepcopy(self.data_repo)
                    self.param_queue.put(to_send)   
                    
        self.iteration_count+=1
        self.ip_to_upload = []
        self.param_to_upload = []
        
        return 0
    
    def set_lock(self):
        pass    
        return 0
            
    def training_check(self):
        ''' Check Existing network against local training data'''
        
        correct = 0
        for idx in range(self.loader.training_size//config['batch_size']):
            client_idx = idx*config['batch_size']
            image_data  = self.loader.load_batch(batch_size=config['batch_size'],index=client_idx,mode='train')
            
            ip          = torch.Tensor(image_data['input']).reshape(config['batch_size'],1,28,28)
            y_target    = image_data['label']

            neck_op     = self.neck.forward(ip)           
            head_op     = self.head.forward(neck_op).cpu().detach().numpy()
            estimate    = [int(x) for x in np.argmax(head_op,axis=1)]
            correct     += int(np.sum(np.equal(y_target,estimate)))
        #print('                             '*int(self.name),self.name,' Train Acc : ',round(correct/self.loader.training_size,5))
        self.training_accuracy = correct/self.loader.training_size
        
    def testing_check(self,to_add):
        ''' Check Existing network against local testing data'''
        
        correct = 0
        for idx in range(self.loader.testing_size//config['batch_size']):
            client_idx = idx*config['batch_size']
            image_data  = self.loader.load_batch(batch_size=config['batch_size'],index=client_idx,mode='test')
            
            ip          = torch.Tensor(image_data['input']).reshape(config['batch_size'],1,28,28)
            y_target    = image_data['label']

            neck_op     = self.neck.forward(ip)           
            head_op     = self.head.forward(neck_op).cpu().detach().numpy()
            estimate    = [int(x) for x in np.argmax(head_op,axis=1)]
            correct     += int(np.sum(np.equal(y_target,estimate)))
            
        self.testing_accuracy = correct/self.loader.testing_size
        #np.save("Results/federated_system/"+config['experiment_name']+'/'+config['experiment_name']+to_add+self.name,self.testing_accuracy)
    
    def run(self):
        
        print(self.name,' Start')
        self.total_loss =  0
        
        while(self.update_count<(config['iterations'])):
            
            #self.check_lock()
            self.download_params()
            self.forward_pass()
            self.upload_params()
            if(self.update_count%100==0):
                self.training_check()
                self.testing_check(None)
                print(self.name,self.update_count,' Train Acc : ',round(self.training_accuracy,5),' Test Acc : ',round(self.testing_accuracy,5))
                self.train_log.append(self.training_accuracy)
                self.test_log.append(self.testing_accuracy)
            #self.set_lock()
        
        ''' Exit process '''
        np.save("Results/federated_system/"+config['experiment_name']+'/'+self.name+'_testing_accuracy'+self.name,np.array(self.test_log))
        np.save("Results/federated_system/"+config['experiment_name']+'/'+self.name+'_training_accuracy'+self.name,np.array(self.train_log))
        queues = [self.param_queue,self.model_queue]
        
        for q in queues:
            while((q.qsize())):
                try: v = q.get(timeout=1)
                except: pass
            
        print(self.name,' Done')
        return 1
            
#x.grad = torch.autograd.Variable(torch.Tensor([5]))
