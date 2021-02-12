# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:00:39 2020

@author: satya
"""

import numpy as np
import matplotlib.pyplot as plt

class DataManager:
    
    def __init__(self,path):
        
        self.accuracies = dict()
        self.losses     = dict()
        self.path       = path
        
    def plot_neck_losses(self,neck):
        
        neck_losses = neck.losses
        plt.plot(neck_losses)
        plt.savefig(self.path+'server_losses')
		
    def plot_head_accuracies(self,head):
        
        accuracy = head.accuracy
        plt.plot(accuracy)
        plt.savefig(self.path+'head_accuracies')