# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:12:53 2020

@author: satya
"""

import time
import yaml
        
from femnist_dataloader import Dataloader
from cnn_head import CNN_Head
from cnn_neck import CNN_Neck
from cnn_server import Server
from cnn_client import Client
from data_manager import DataManager

from utilities import freeze_layers
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import csv
import os
import pickle

import multiprocessing as mp

import queue

''' MULTIPOROCESSING '''

if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.autograd.set_detect_anomaly(True)

  
    ''' Create Experiment Specific Files and Parameters '''
    
    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    mode = 'cuda'
    exp_path = "Results/federated_system/"+config['experiment_name']+'/'
    if(not(os.path.exists(exp_path))):
        os.mkdir(exp_path)
    with open(exp_path+config['experiment_name']+'_config.csv','w',newline='') as file:
        writer = csv.writer(file)
        for key in config:
            writer.writerow([str(key).ljust(25)+' : '+str(config[key])])
    
    file_indices = [i for i in range(config['num_sets'])]
    #random.shuffle(file_indices)
    client_slice = len(file_indices)//config['num_clients']
    
    ''' Objects to Hold Data Generated '''
    dm = DataManager(exp_path)
    
    ''' Create Upload Schedules '''
    
    upload_schedule = []
    
    for i in range(config['iterations']*2):
        clients = [str(i) for i in range(config['num_clients'])]
        random.shuffle(clients)
        client_ids = clients[:int(config['upload_fraction']*config['num_clients'])]
        upload_schedule.append(client_ids)
        
    with open(exp_path+"upload_schedule.txt", 'w') as f:
        for client_selection in upload_schedule:
            f.write(str(client_selection) + '\n')
        
    ''' Instantiate Global Data Repo '''

    param_queue =  mp.Queue(maxsize=config['num_clients']*2)
    model_queue =  mp.Queue(maxsize=10)
    update_queue = mp.Queue(maxsize=1)
    
    ''' Generate initial Weights for Networks '''
    
    neck_template = CNN_Neck(mode)
    head_template = CNN_Head(mode)
    
    neck_weights = neck_template.get_weights()
    head_weights = head_template.get_weights()
    
    ''' Instantiate Head and Neck Networks '''
    
    server_neck = CNN_Neck(mode)
    server_head = CNN_Head(mode)
    
    server_neck.load_weights(neck_weights)
    server_head.load_weights(head_weights)
    
    server = Server('server',
                    server_neck,
                    upload_schedule,
                    model_queue,
                    param_queue, 
                    head_network = server_head,
                    data_manager = dm)
    
    ''' Instantiate Server and Clients '''
    
    threads = []
    dataset_split = []
    
    for i in range(config['num_clients']):
        
        head = CNN_Head(mode)
        neck = CNN_Neck(mode)
        
        neck.load_weights(neck_weights)
        head.load_weights(head_weights)
    
        loader = Dataloader(file_indices,[i*(client_slice),min((i+1)*(client_slice),35)])    
        loader.load_training_dataset()
        loader.load_testing_dataset()
        threads.append(Client(str(i),
                              neck,
                              head,
                              upload_schedule,
                              model_queue,
                              param_queue,
                              loader,
                              data_manager = dm))
        
        dataset_split.append([i*(client_slice),min((i+1)*(client_slice),35)])
        
    with open(exp_path+"file_indices.txt", 'w') as f:
        for file in dataset_split:
            f.write(str(file) + '\n')
            
    processes = []
    for idx in range(len(threads)):
        x = mp.Process(target=threads[idx].run)
        processes.append(x)
        
    y = mp.Process(target=server.run)
    for t in processes:
        t.start()
    y.start()
        
    y.join()    
    for t in processes:
        t.join()
 
    ''' COMPLETE '''
    
    end_time = int(time.time())
    
    print('Completed')
    
    np.save(exp_path+config['experiment_name']+'_neck_losses',server.losses)
    torch.save(server.neck_network.network.state_dict(),exp_path+config['experiment_name']+'_neck_network')
    '''with open(exp_path+config['experiment_name']+'_neck_network', 'wb') as network_file:
        pickle.dump(server.neck_network, network_file)'''
    
    for idx in range(len(threads)):
            
        client = threads[idx]
        plt.plot(client.losses,label=client.name)
        np.save(exp_path+config['experiment_name']+'_'+client.name+'_losses',client.losses)
        torch.save(client.head.network.state_dict(),exp_path+config['experiment_name']+'_'+client.name+'_head_network')
        torch.save(client.neck.network.state_dict(),exp_path+config['experiment_name']+'_'+client.name+'_neck_network') 
        '''with open(exp_path+config['experiment_name']+'_'+client.name+'_head_network', 'wb') as network_file:
            pickle.dump(client.head, network_file)
        with open(exp_path+config['experiment_name']+'_'+client.name+'_neck_network', 'wb') as network_file:
            pickle.dump(client.neck, network_file)'''
            
    plt.grid()
    plt.legend()
    plt.title('Losses')
    plt.savefig(exp_path+'losses')
    plt.close()
    
    for idx in range(len(threads)):
            
        client = threads[idx]
        plt.plot(client.train_log,label=client.name+' Train',alpha=0.5)
        plt.plot(client.test_log,label=client.name+' Test',alpha=0.5)
    plt.grid()
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(exp_path+'accuracies')
    plt.close()
    
    for idx in range(len(threads)):
        client = threads[idx]
        plt.plot(client.losses,label=client.name)
    
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.title('Losses - Log')
    plt.savefig(exp_path+'losses_log')
    plt.close()
    
    plt.plot(server.losses,label='Server')
    plt.grid()
    plt.legend()
    plt.title('Server Losses')
    plt.savefig(exp_path+'server_losses')
    plt.close()
    
            
    with open(exp_path+config['experiment_name']+'_testing_accuracies.csv','w',newline='') as file:
        writer = csv.writer(file)
        for cidx in range(config['num_clients']):
            client = threads[cidx]
            client_idx = cidx
            avg_acc = 0
            for idx2 in range(config['num_clients']):
   
                loading_client = threads[idx2]
                client.head.hook_mode='test'
            
                loading_client.loader.load_testing_dataset()
                correct = 0
                
                for idx in range(loading_client.loader.testing_size//config['batch_size']):
                    client_index = idx*config['batch_size']
                    image_data  = loading_client.loader.load_batch(batch_size=config['batch_size'],index=client_idx,mode='test')
                
                    ip          = torch.Tensor(image_data['input']).reshape(config['batch_size'],1,28,28)
                    y_target    = image_data['label']
    
                    neck_op     = client.neck.forward(ip)           
                    head_op     = client.head.forward(neck_op).cpu().detach().numpy()
                    estimate    = [int(x) for x in np.argmax(head_op,axis=1)]
                    correct     += int(np.sum(np.equal(y_target,estimate)))

                avg_acc+=correct/loading_client.loader.testing_size
                print("Client ",client_idx," with Test Set of ",idx2," : Test Accuracy : ",correct/loading_client.loader.testing_size)
                writer.writerow([client_idx,idx2,correct/loading_client.loader.testing_size])
            print("Client ",client_idx,': Average Accuracy : ',avg_acc/config['num_clients'])
            print()