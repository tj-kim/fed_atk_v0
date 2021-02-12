# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:01:42 2020

@author: satya
"""

'''

DATALOADER CLASS

Convert data downloaded from Leaf to consumable format

Images (.png) --> Numpy

'''

import glob
import numpy as np
from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt
import os

class Dataloader:
    
    def __init__(self,num_classes=None,get_image=False):
        
        self.repo = {'input':[],
                     'label':[]}
        self.classes    = None
        self.num_dirs   = num_classes
        self.get_image  = get_image
        
    def set_path(self,path):
        
        self.path = path
        
        if(self.num_dirs==None):
            self.dirs = os.listdir(path)
        else:
            self.dirs = np.random.choice(os.listdir(path),self.num_dirs)
            print('Classes Selected : ',self.dirs)
        
    def get_classes(self):
        self.classes = list(self.dirs)
        return self.classes
        
    def empty_repo(self):
        self.repo = {'input':[],
                     'label':[]}
    
    def load_images(self,batch_size=1):
        
        self.image_data = []
        for idx in range(batch_size):
            
            path = self.path
            
            dir     = np.random.choice(self.dirs)
            path    += '/'+dir
            
            self.repo['label'].append(self.classes.index(str(dir)))
            
            dirs    = os.listdir(path)
            for dir in dirs:
                if('.mit' in dir):
                    dirs.remove(dir)
            dir = np.random.choice(dirs)
            path += '/'+dir
            
            dirs    = os.listdir(path)
            dir     = np.random.choice(dirs)
            path    += '/'+dir
            
            if(self.get_image):
                self.repo['input'].append(image.imread(path)[:,:,0])
            else:
                self.repo['input'].append(image.imread(path)[:,:,0].flatten())
                
        
               
        return self.repo