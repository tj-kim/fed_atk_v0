# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:23:57 2020

@author: satya
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

   
#plt.figure(figsize=(20,10)) 
def plot_accuracies(i):
    for i in range(8):
        w = 3
        x_pos = np.array([1])*21
        plt.figure(figsize=(5,10))
        
        exp = ['random','average','individual']
        for mode in exp:
            
            train_filename = 'Results/'+mode+'_head_networks/'+str(i)+'_training_accuracy'+str(i)+'.npy'
            test_filename = 'Results/'+mode+'_head_networks/'+str(i)+'_testing_accuracy'+str(i)+'.npy'
            
            train = np.load(train_filename)[-1]
            test = np.load(test_filename)[-1]
            
            
            #x_pos+=w
            #plt.bar(x_pos,train,width=w,label=mode+' '+str(i)+' Train',alpha=0.5)
            
            x_pos+=w
            plt.bar(x_pos,test,width=w,label=mode+' '+str(i)+' Test',alpha=0.5)
            
            plt.ylim(0.7,0.85)
            plt.grid()
            plt.legend()
            
        plt.show()
            
        '''print(i)
        print(train[-1])
        print(test[-1])
        print()'''
    return train,test
        
train,test = plot_accuracies(7)
    

'''exp = 'random_head_networks' 
i = 4
train_filename = 'Results/'+exp+'/'+exp+'_'+str(i)+'_head_network'

x = torch.load(train_filename)
weights = x['0.weight'].cpu().detach().numpy()

plt.figure(figsize=(40,80))
plt.imshow(weights,cmap='hot', interpolation='nearest')
plt.show()
print(np.max(weights))
print(np.min(weights))'''