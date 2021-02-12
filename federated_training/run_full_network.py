# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:23:38 2020

@author: satya
"""

from femnist_dataloader import Dataloader
from full_cnn import Full_CNN
import numpy as np
import torch
import matplotlib.pyplot as plt

exp_name = 'cuda'#input('Please Enter Experiment Name : ')
mode 		= 'cuda'#input('Please Enter Mode : ')
loader = Dataloader()    
loader.load_training_dataset()
loader.load_testing_dataset()

full_network        = Full_CNN(mode)
full_network_losses = []
test_score = []

batch_size = 256

for i in range(10000):
    
    image_data = loader.load_batch(batch_size=batch_size)
    
    ip          = torch.Tensor(image_data['input']).reshape(batch_size,1,28,28)
    if(mode=='cuda'):
        y_target    = torch.Tensor(image_data['label']).type(torch.LongTensor).cuda()
    else:
        y_target    = torch.Tensor(image_data['label']).type(torch.LongTensor)
		
    op = full_network.forward(ip)
    
    loss = full_network.loss_critereon(op,y_target)
    
    full_network_losses.append(loss.cpu().detach().numpy())
    
    if(i%100==0):
        correct = 0

        for j in range(loader.testing_size):
			
            image_data = loader.load_batch(batch_size=1,mode='test')
			
            ip          = torch.Tensor(image_data['input']).reshape(1,1,28,28)
            if(mode=='cuda'):
                y_target    = torch.Tensor(image_data['label']).type(torch.LongTensor).cuda()
            else:
                y_target    = torch.Tensor(image_data['label']).type(torch.LongTensor).cuda()

            op          = full_network.forward(ip).cpu().detach().numpy()[0]
            estimate    = int(np.argmax(op))
			
            correct+=int(y_target==estimate)
            
        test_score.append(correct/loader.testing_size)
        
        print("Iteration ",i," | Loss : ",loss.cpu().detach().numpy())
        print("Test Accuracy : ",correct/loader.testing_size)
        print()
               
    full_network.optimizer.zero_grad() 
    loss.backward()
    full_network.optimizer.step()	
    
    # Gradient processes if any
    
    x = full_network.get_activations_gradient()
    
		
loader.delete_training_dataset()

print('Testing Accuracy : ',correct/loader.testing_size)

plt.title("Full CNN Losses")
plt.plot(full_network_losses)
plt.grid()
plt.savefig(exp_name+'_fullnetwork')
plt.show()
plt.clf()

plt.title("Testing Accuracy")
plt.plot(test_score)
plt.grid()
plt.savefig(exp_name+'_testscores')
plt.show()
plt.clf()
