# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 20:05:47 2020

@author: satya
"""

from femnist_dataloader import Dataloader
from cnn_head import CNN_Head
from cnn_neck import CNN_Neck
from cnn_server import Server
from cnn_client import Client

import matplotlib as plt
import numpy as np
import torch
import time

from torch.autograd import Variable
from copy import deepcopy
from utilities import freeze_layers
import threading
import yaml
import torch.nn as nn
import random
import csv
import pickle

with open(r'config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
mode = 'cuda'
    
experiment_name = input("Please Enter Experiment Name : ")

config['experiment_name'] = experiment_name

exp_path = "Results/federated_system/"+config['experiment_name']+'/'

''' Create Upload Schedules '''

upload_schedule = []

for i in range(config['iterations']*2):
    clients = [str(i) for i in range(config['num_clients'])]
    random.shuffle(clients)
    client_ids = clients[:5]
    upload_schedule.append(client_ids)
    
with open(exp_path+"upload_schedule.txt", 'w') as f:
    for client_selection in upload_schedule:
        f.write(str(client_selection) + '\n')
    
''' Instantiate Server and Clients '''
''' Instantiate Global Data Repo '''

data_repo = {'from_server':dict(),
             'from_clients':dict(),
             'update_count':0,
             'now_running':dict(),
             'lock':False}

file_indices = [i for i in range(35)]
random.shuffle(file_indices)

client_slice = len(file_indices)//config['num_clients']
for i in range(config['num_clients']):
    data_repo['from_clients'][str(i)] = {'inputs':[],'params':[]}
    data_repo['now_running'][str(i)] = True
data_repo['now_running']['server'] = False

threads =[]
neck_path = exp_path+experiment_name+'_0_neck_network'
for i in range(config['num_clients']):
    
    head = CNN_Head('cuda')
    neck = CNN_Neck('cuda')
    # Load Networks
    head_path = exp_path+experiment_name+'_'+str(i)+'_head_network'
    #neck_path = exp_path+experiment_name+'_neck_network'
    
    head.network.load_state_dict(torch.load(head_path))
    neck.network.load_state_dict(torch.load(neck_path))
    
    '''with open(head_path, 'rb') as head_file:
        head = pickle.load(head_file)
    with open(neck_path, 'rb') as neck_file:
        neck = pickle.load(neck_file)'''
        
    loader = Dataloader([i*(client_slice),min((i+1)*(client_slice),35)])    
    threads.append(Client(str(i),neck,head,upload_schedule,data_repo,loader))


with open(exp_path+config['experiment_name']+'_testing_accuracies.csv','w',newline='') as file:
    writer = csv.writer(file)
    for idx in range(config['num_clients']):

        client = threads[idx]
        client.head.hook_mode='test'
        client_idx = idx
        
        client.loader.load_training_dataset()
        client.loader.load_testing_dataset()
        correct = 0
			
        for idx in range(client.loader.testing_size//config['batch_size']):
            client_index = idx*config['batch_size']
            image_data  = client.loader.load_batch(batch_size=config['batch_size'],index=client_idx,mode='test')
			
            ip          = torch.Tensor(image_data['input']).reshape(config['batch_size'],1,28,28)
            y_target    = image_data['label']

            neck_op     = client.neck.forward(ip)           
            head_op     = client.head.forward(neck_op).cpu().detach().numpy()
            estimate    = [int(x) for x in np.argmax(head_op,axis=1)]
            correct     += int(np.sum(np.equal(y_target,estimate)))
            
        print("Client ",client_idx," : Test Accuracy : ",correct/client.loader.testing_size)