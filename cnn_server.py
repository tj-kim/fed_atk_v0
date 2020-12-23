# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:49:22 2020

@author: satya
"""

import torch
import numpy as np
import time
import threading
import yaml
import copy
import multiprocessing as mp
import csv
from torch.autograd import Variable

with open(r'config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

class Server():
    
    def __init__(self,name,neck_network,update_schedule,model_queue,param_queue,head_network=None,data_manager=None):
        
        self.name = name
        self.neck_network         = neck_network
        self.head_network         = head_network
        
        self.data_repo            = {'model':None,
                                     'time':0}
        self.client_ids           = [str(i) for i in range(config['num_clients'])]
        self.num_clients          = len(self.client_ids)
        
        #self.data_repo['model']      = copy.deepcopy(self.neck_network)
        self.data_repo['time']  = 1
        self.data_repo['update_count']  = 0
        self.data_repo['training_complete']  = False
        
        self.batch_size = config['server_batch_size']
        self.training_iterations = config['server_iterations']
        
        self.aggregated_params = {'inputs':[],'params':[]}
        self.losses = []
        self.update_schedule = update_schedule
        
        self.aggregated_params = {'inputs':[],'params':[]}
        self.inputs = []
        self.params = []
        
        self.model_queue = model_queue
        self.param_queue = param_queue
        self.head_network.pretrain = True
        
        self.client_distributions = np.zeros((config['num_clients'],63))
        self.data_manager = data_manager
        self.losses = []
        
        self.has_aggregated_parameters  = False
        self.has_updated_model          = False
        self.has_distributed_model      = False
        
    def check_lock(self):
        pass
        return 0
    
    def reset_flags(self):
        
        self.has_aggregated_parameters  = False
        self.has_updated_model          = False
        self.has_distributed_model      = False
        
    def aggregate_parameters(self):
        ''' Collect Data uploaded by clients from Queue '''
        
        self.inputs     = []
        self.params     = []
        self.ids        = []
        self.head_networks    = dict()
        
        # Keep querying until enough data is available
        while(self.param_queue.qsize()<int(config['num_clients']*config['upload_fraction'])):    
            time.sleep(0.5)
            pass

        while(self.param_queue.qsize()>0):
            client_data = self.param_queue.get()
            self.inputs.append(list(client_data['inputs']))
            self.params.append(list(client_data['params']))
            self.ids.append(list(client_data['ids']))
            self.head_networks[client_data['ids'][0]] = client_data['network']

        self.has_aggregated_parameters = True
      
    def loss_function(self,x,y):
        loss = y-x
        return loss
        
    def update_model(self):
        ''' Train Global model to be distributed to Clients later '''
        
        if(self.has_aggregated_parameters):
            iteration_repo = {'inputs':[],'params':[],'ids':[]}
                
            '''for idx in range(len(self.inputs)):
                iteration_repo['inputs'].append([self.inputs[idx]])
                iteration_repo['params'].append([self.params[idx]])
                iteration_repo['ids'].append([self.ids[idx]])'''
                
            iteration_repo['inputs'] = self.inputs
            iteration_repo['params'] = self.params
            iteration_repo['ids'] = self.ids
            
            total_loss = 0
            
            '''for idx in range(len(config['head_architecture'])):
                weights = torch.zeros_like(self.head_networks[self.ids[0][0]][0][idx])
                biases  = torch.zeros_like(self.head_networks[self.ids[0][0]][1][idx])
                for key in self.head_networks:
                    weights+=(torch.tensor(self.head_networks[key][0][idx]))
                    biases+=(torch.tensor(self.head_networks[key][1][idx]))
                self.head_network.network[idx].weight.data.copy_(weights)
                self.head_network.network[idx].bias.data.copy_(biases)'''
            for i in range(config['server_iterations']):
                
                '''batch_idx = np.random.randint(0,len(iteration_repo['params']),config['server_batch_size'])
                
                ip      = torch.Tensor([iteration_repo['inputs'][idx] for idx in batch_idx]).reshape(config['server_batch_size'],1,28,28)
                target  = torch.Tensor([iteration_repo['params'][idx] for idx in batch_idx]).cuda()'''
                
                '''ip      = torch.Tensor(iteration_repo['inputs']).reshape(len(iteration_repo['inputs']),1,28,28)
                target  = torch.Tensor(iteration_repo['params']).reshape(len(iteration_repo['inputs']),1,1600).cuda()
                neck_op = self.neck_network.forward(ip)  
                neck_op = Variable(neck_op,requires_grad=True)              
                target  = neck_op + target
                target  = target.detach()
                
                loss    = self.loss_function(neck_op,target)
                for element in loss[0]:
                    element[0].backward()
                self.neck_network.optimizer.step()
                self.neck_network.optimizer.zero_grad()'''

                idx = np.random.randint(0,len(iteration_repo['params']))
                ip      = torch.Tensor(iteration_repo['inputs'][idx]).reshape(len(iteration_repo['inputs'][idx]),1,28,28)
                target  = torch.Tensor(iteration_repo['params'][idx]).type(torch.LongTensor).cuda()
                
                client_idx = iteration_repo['ids'][idx][0]
                for idx in range(len(config['head_architecture'])):
                    self.head_network.network[idx].weight.data.copy_(torch.tensor(self.head_networks[client_idx][0][idx]))
                    self.head_network.network[idx].bias.data.copy_(torch.tensor(self.head_networks[client_idx][1][idx]))
                    
                
                neck_op = self.neck_network.forward(ip)  
                neck_op = Variable(neck_op,requires_grad=True)              
                
                head_op = self.head_network.forward(neck_op)
                
                loss    = self.head_network.loss_critereon(head_op,target)
                loss.backward()
                self.neck_network.optimizer.step()
                #self.head_network.optimizer.step()
                self.neck_network.optimizer.zero_grad() 
                #self.head_network.optimizer.zero_grad()
                
            total_loss+=loss.cpu().detach().numpy()
            
            self.losses.append(total_loss/config['server_iterations'])
            
            self.has_updated_model = True
        else:
            pass
            
    def distribute_model(self,complete=False):
        ''' Push Model updates to Clients '''
       
        # Flush Queue
        while(self.model_queue.qsize()>0):self.model_queue.get()
        
        # Collect Weights
        update_network = [] 
        for idx in range(len(self.neck_network.network)):
            
            weights = self.neck_network.network[idx].weight.data.cpu().detach().numpy()
            bias    = self.neck_network.network[idx].bias.data.cpu().detach().numpy()
            update_network.append([weights,bias])
            
        self.data_repo['neck_model']     = update_network
        
        update_network = [] 
        for idx in range(len(self.head_network.network)):
            
            weights = self.head_network.network[idx].weight.data.cpu().detach().numpy()
            bias    = self.head_network.network[idx].bias.data.cpu().detach().numpy()
            update_network.append([weights,bias]) 
            
        self.data_repo['head_model']     = update_network
        
        self.data_repo['time']          = int(time.time())
        self.data_repo['complete']      = complete
        
        self.data_repo['update_count'] += 1
        if(complete):
            self.data_repo['update_count'] = config['iterations']+10
        # Uplaod To Queue
        for i in range(10):
            self.model_queue.put(self.data_repo)
         
        
    def set_lock(self):
        pass
        return 0
                
    def check_client_distances(self):
        
        self.client_distances = np.zeros((config['num_clients'],config['num_clients']))
        for i in range(config['num_clients']):
            tot_i = np.sum(self.client_distributions[i])
            for j in range(config['num_clients']):
                tot_j = np.sum(self.client_distributions[j])
                
                i_dist = self.client_distributions[i]/(tot_i+1e-6)
                j_dist = self.client_distributions[j]/(tot_j+1e-6)
                self.client_distances[i,j] = round(np.sum(np.abs(i_dist-j_dist)),3)
                    
        print(self.client_distances)
                    
            

    def run(self):
        print(self.name,'Start')
        self.distribute_model()
        
        while(self.data_repo['update_count']<config['iterations']):
            #print(self.name,self.data_repo['update_count'])
            self.reset_flags()
            
            self.aggregate_parameters()
            self.update_model()
            self.distribute_model()
        
        self.distribute_model(complete=True)
        #np.save("Results/federated_system/"+config['experiment_name']+'/'+config['experiment_name']+'server_losses',self.losses)
            
        print(self.name,' Done')
        return 1
            
        
            
        
        