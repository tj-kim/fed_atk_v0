# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:03:10 2020

@author: satya
"""

import numpy as np
import json
import os
import time
import gc
import yaml
from pathlib import Path


class Dataloader:
    
    def __init__(self,file_order,file_range,file_path='federated_training/leaf/data/femnist/data'):
        self.train_file_path = file_path+'/train/'
        self.test_file_path = file_path+'/test/'
        self.repo = {'input':[],'label':[]}
        self.train_dataset  = None
        self.test_dataset   = None
        self.file_range  = file_range
        self.training_size = 0
        self.file_order = file_order
        
        with open(r'configs/config.yaml') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        
    def load_training_dataset(self):
        
        self.train_dataset = {'users':[],'user_data':dict()}
        
        files = os.listdir(self.train_file_path)
        files = [files[i] for i in self.file_order][self.file_range[0]:self.file_range[1]]
        
        for json_file in files:
            print('Loading ',json_file)	

            json_file_path = self.train_file_path+json_file
            
            with open(json_file_path,'r') as f:
                data = json.load(f)
            writers = data['users']
            
            for writer in writers:
                if(not(writer in self.train_dataset['users'])):
                    self.train_dataset['users'].append(writer)
                    self.train_dataset['user_data'][writer] = dict()
                    self.train_dataset['user_data'][writer]['x'] = []
                    self.train_dataset['user_data'][writer]['y'] = []
                
                self.train_dataset['user_data'][writer]['x']+=data['user_data'][writer]['x']
                self.train_dataset['user_data'][writer]['y']+=data['user_data'][writer]['y']
                
                self.training_size += len(self.train_dataset['user_data'][writer]['y'])
			
            del(data)
            gc.collect()

    def load_testing_dataset(self):
        
        self.test_dataset = {'users':[],'user_data':dict()}
        self.test_dataset['user_data'] = {'x':[],'y':[]}
        
        files = os.listdir(self.test_file_path)
        files = [files[i] for i in self.file_order][self.file_range[0]:self.file_range[1]]
        
        for json_file in files:
		
            json_file_path = self.test_file_path+json_file
            
            with open(json_file_path,'r') as f:
                data = json.load(f)
            writers = data['users']
            
            for writer in writers:
                if(not(writer in self.test_dataset['users'])):
                    self.test_dataset['users'].append(writer)
                
                self.test_dataset['user_data']['x']+=data['user_data'][writer]['x']
                self.test_dataset['user_data']['y']+=data['user_data'][writer]['y']
            
        self.testing_size = len(self.test_dataset['user_data']['y'])
            
    def delete_training_dataset(self):
        
        del self.train_dataset
            
    def load_batch(self,batch_size,mode='train',index = 0):
    
        self.batch = {'input':[],'label':[]}
        
        for i in range(batch_size):
            
            if(mode=='train'):
                writer = np.random.choice(self.train_dataset['users'])
                random_idx = np.random.randint(0,len(self.train_dataset['user_data'][writer]['y']))
                
                self.batch['input'].append(self.train_dataset['user_data'][writer]['x'][random_idx])
                self.batch['label'].append(self.train_dataset['user_data'][writer]['y'][random_idx])
                
            if(mode=='test'):
                
                if(index=='all'):
                    self.batch['input'].append(self.test_dataset['user_data']['x'])
                    self.batch['label'].append(self.test_dataset['user_data']['y'])
                else:
                    index+=1
                    self.batch['input'].append(self.test_dataset['user_data']['x'][index])
                    self.batch['label'].append(self.test_dataset['user_data']['y'][index])
				
        return self.batch