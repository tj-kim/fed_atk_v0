import torch
import torch.nn as nn

from collections import OrderedDict 
import itertools

from federated_training.femnist_dataloader import Dataloader
from federated_training.cnn_head import CNN_Head
from federated_training.cnn_neck import CNN_Neck
from federated_training.cnn_server import Server
from federated_training.cnn_client import Client
from federated_training.data_manager import DataManager
from federated_training.utils import cuda, where
from federated_training.utilities import freeze_layers

from cw_attack.cw import *

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import csv
import os
import pickle
from torch.autograd import Variable
import copy

class Personalized_NN(nn.Module):
    """
    Pytorch NN module that combines head and neck weights for a layered based sharing system
    For training federated learning neural network.
    """
    
    def __init__(self, head_network, neck_network):
        
        # Init attributes
        super(Personalized_NN, self).__init__()
        self.head = head_network
        self.neck = neck_network
        self.criterion = nn.NLLLoss()
        
        # test_acc attributes
        self.orig_test_acc = None
        self.adv_test_acc = None
        
        # Split Test att
        self.orig_test_acc_robust = None
        self.orig_output_sim_robust = None
        self.orig_test_acc_adv = None
        self.orig_output_sim_adv = None
        
        
        self.orig_output_sim = None
        self.adv_output_sim = None
        
        self.orig_target_achieve = None
        self.adv_target_achieve = None
        
        # Log correct vs. Incorrect Indices
        self.adv_indices = None # tensor List of indices from batch that are fooled
        self.robust_indices = None # List of indices from batch robust
        
    def forward(self,x):
        
        if torch.cuda.is_available():
            x = x.cuda()
        
        x = self.neck.forward(x)
        x = self.head.forward(x)
        
        return x
    
    def forward_transfer(self, x_orig, x_adv, y_orig, y_adv,
                         true_labels, target, print_info = False):
        """
        Assume that input images are in pytorch tensor format
        """
        self.eval()
        
        # Cuda Availability
        if torch.cuda.is_available():
            (y_orig, y_adv) = (y_orig.cuda(), y_adv.cuda())
        
        batch_size = y_orig.shape[0]
        
        # Forward Two Input Types
        h_adv = self.forward(x_adv)
        h_orig = self.forward(x_orig)
        h_adv_category = torch.argmax(h_adv,dim = 1)
        h_orig_category = torch.argmax(h_orig,dim = 1)
        
        # Record Different Parameters
        self.orig_test_acc = (h_orig_category == true_labels).float().sum()/batch_size
        self.adv_test_acc = (h_adv_category == true_labels).float().sum()/batch_size
        
        self.orig_output_sim = (h_orig_category == y_orig).float().sum()/batch_size
        self.adv_output_sim = (h_adv_category == y_adv).float().sum()/batch_size
        
        self.orig_target_achieve = (h_orig_category == target).float().sum()/batch_size
        self.adv_target_achieve = (h_adv_category == target).float().sum()/batch_size
        
        # Record based on indices
        self.adv_indices = h_adv_category == target
        self.robust_indices = h_adv_category != target
        
        # Record split losses
        self.orig_test_acc_robust = (h_orig_category[self.robust_indices] == true_labels[self.robust_indices]).float().sum()/h_orig_category[self.robust_indices].shape[0]
        self.orig_output_sim_robust = (h_orig_category[self.robust_indices] == y_orig[self.robust_indices]).float().sum()/h_orig_category[self.robust_indices].shape[0]
        
        self.orig_test_acc_adv = (h_orig_category[self.adv_indices] == true_labels[self.adv_indices]).float().sum()/h_orig_category[self.adv_indices].shape[0]
        self.orig_output_sim_adv = (h_orig_category[self.adv_indices] == y_orig[self.adv_indices]).float().sum()/h_orig_category[self.adv_indices].shape[0]
            

        # Print Relevant Information
        if print_info:
            print("---- Attack Transfer:", "----\n")
            print("         Orig Test Acc:", self.orig_test_acc.item())
            print("          Adv Test Acc:", self.adv_test_acc.item())
            print("Orig Output Similarity:", self.orig_output_sim.item())
            print(" Adv Output Similarity:", self.adv_output_sim.item())
            print("       Orig Target Hit:", self.orig_target_achieve.item())
            print("        Adv Target Hit:", self.adv_target_achieve.item())
            
        return (h_adv, h_orig, h_adv_category, h_orig_category)


class Adv_NN(Personalized_NN):
    
    def __init__(self, head_network, neck_network, dataloader):
        
        # Init attributes
        super(Adv_NN, self).__init__(head_network, neck_network)
        
        self.dataloader = dataloader
        # self.criterion = nn.NLLLoss()
        
        # Attack outputs
        self.x_orig = None
        self.x_adv = None
        self.y_orig = None
        
        self.softmax_orig = None
        self.output_orig = None
        self.softmax_adv = None
        self.output_adv = None 
        
        
    def i_fgsm(self, atk_params, print_info=False, mode = 'test'):
        """
            Perform IFSGM attack on a randomly sampled batch 
            All attack params and batch sizes are defiend in atk_params
        """
        self.eval()
        
        # Import attack parameters
        batch_size = atk_params.batch_size
        target= atk_params.target
        eps= atk_params.eps
        alpha= atk_params.alpha
        iteration= atk_params.iteration
        x_val_min= atk_params.x_val_min
        x_val_max= atk_params.x_val_max
        
        # Load data to perturb
    
        image_data = self.dataloader.load_batch(batch_size, mode=mode)
        self.x_orig  = torch.Tensor(image_data['input']).reshape(batch_size,1,28,28)
        self.y_orig = torch.Tensor(image_data['label']).type(torch.LongTensor)
        
        if torch.cuda.is_available():
            self.y_orig = self.y_orig.cuda()
        
        self.target = target
        
        self.x_adv = Variable(self.x_orig, requires_grad=True)
        
        for i in range(iteration):
            
            h_adv = self.forward(self.x_adv)
            
            # Loss function based on target
            if target > -1:
                target_tensor = torch.LongTensor(self.y_orig.size()).fill_(target)
                target_tensor = Variable(cuda(target_tensor, self.cuda), requires_grad=False)
                cost = self.criterion(h_adv, target_tensor)
            else:
                cost = -self.criterion(h_adv, self.y_orig)

            self.zero_grad()

            if self.x_adv.grad is not None:
                self.x_adv.grad.data.fill_(0)
            cost.backward()

            self.x_adv.grad.sign_()
            self.x_adv = self.x_adv - alpha*self.x_adv.grad
            self.x_adv = where(self.x_adv > self.x_orig+eps, self.x_orig+eps, self.x_adv)
            self.x_adv = where(self.x_adv < self.x_orig-eps, self.x_orig-eps, self.x_adv)
            self.x_adv = torch.clamp(self.x_adv, x_val_min, x_val_max)
            self.x_adv = Variable(self.x_adv.data, requires_grad=True)

        self.post_attack(batch_size = batch_size, print_info = print_info)
    
    def CW_attack(self, attack_params, print_info=False, mode='test'):
        
        self.eval()
        
        # copy attack parameters
        batch_size = attack_params.batch_size
        target= attack_params.target
        confidence= attack_params.confidence
        optimizer_lr= attack_params.optimizer_lr
        iteration= attack_params.iteration
        x_val_mean = attack_params.x_val_mean
        x_val_std= attack_params.x_val_std
        
        # Calculate inputs box
        inputs_box = (min((0 - m) / s for m, s in zip(x_val_mean, x_val_std)),
                           max((1 - m) / s for m, s in zip(x_val_mean, x_val_std)))
        
        image_data = self.dataloader.load_batch(batch_size, mode=mode)
        self.x_orig  = torch.Tensor(image_data['input']).reshape(batch_size,1,28,28)
        self.y_orig = torch.Tensor(image_data['label']).type(torch.LongTensor)
        self.target = target
        
        if torch.cuda.is_available():
            self.y_orig = self.y_orig.cuda()
            x_orig = self.x_orig.cuda()
        else:
            x_orig = self.x_orig
            
        
        if target > -1:
            targeted = True
            if torch.cuda.is_available():
                targets = torch.ones(self.x_orig.size(0), dtype = torch.long).cuda() * target
            else:
                targets = torch.ones(self.x_orig.size(0), dtype = torch.long) * target  
        else:
            targeted = False
            targets = self.y_orig

            
        
        adversary = L2Adversary(targeted=targeted,
                           confidence=confidence,
                           search_steps=iteration,
                           box= inputs_box,
                           optimizer_lr=optimizer_lr)
        
        self.x_adv = adversary(self, x_orig, targets, to_numpy=False)
        
        self.post_attack(batch_size = batch_size, print_info = print_info)
            
    def post_attack(self, batch_size, print_info = False):
        """
        Computes attack success metrics after xadv is generated
        """
                
        self.softmax_orig = self.forward(self.x_orig)
        self.output_orig = torch.argmax(self.softmax_orig,dim=1)
        self.softmax_adv = self.forward(self.x_adv)
        self.output_adv = torch.argmax(self.softmax_adv,dim=1)
        
        # Record accuracy and loss
        self.orig_loss = self.criterion(self.softmax_orig, self.y_orig).item()
        self.adv_loss = self.criterion(self.softmax_adv, self.y_orig).item()
        self.orig_acc = (self.output_orig == self.y_orig).float().sum()/batch_size
        self.adv_acc = (self.output_adv == self.y_orig).float().sum()/batch_size
        
        # Add Perturbation Distance (L2 norm) - across each input
        self.norm = torch.norm(torch.sub(self.x_orig, self.x_adv, alpha=1),dim=(2,3))

        # Print Relevant Information
        if print_info:
            print("---- FGSM Batch Size:", batch_size, "----\n")
            print("Orig Target:", self.y_orig.tolist())
            print("Orig Output:", self.output_orig.tolist())
            print("ADV Output :", self.output_adv.tolist(),'\n')
            print("Orig Loss  :", self.orig_loss)
            print("ADV Loss   :", self.adv_loss,'\n')
            print("Orig Acc   :", self.orig_acc.item())
            print("ADV Acc    :", self.adv_acc.item())
        
        
def load_FLNN(idx, direc, loader = None):
    # Load the corresponding head/neck network in victim nn module 
    
    if torch.cuda.is_available():
        mode = 'cuda'
    else:
        mode = 'not_cuda'
    
    head_nn = CNN_Head(mode)
    neck_nn = CNN_Neck(mode)
    
    # Which network to load and directory
    exp_path = "federated_training/Results/federated_system/" + direc + "/"
    nn_path = exp_path + direc + "_"
    
    # Load pre-trained weights
    head_path = nn_path + str(idx) +"_head_network"
    neck_path = nn_path + str(idx) +"_neck_network"

    """
    print("load_FLNN exppath:", head_path)
    print("load_FLNN exppath:", neck_path)
    print("head_nn \n", head_nn)
    print("neck_nn \n",neck_nn)
    print(head_nn.config)
    print(neck_nn.config)
    """
    
    head = torch.load(head_path)
    neck = torch.load(neck_path)

    head_edit = OrderedDict()
    neck_edit = OrderedDict()

    # Edit the ordered_dict key names to be torch compatible
    for key in head.keys():
        head_edit["network."+key] = head[key]

    for key in neck.keys():
        neck_edit["network."+key] = neck[key]

    head_nn.load_state_dict(head_edit)
    neck_nn.load_state_dict(neck_edit)
    
    if loader is None:
        return Personalized_NN(head_nn,neck_nn)
    else:
        return Adv_NN(head_nn,neck_nn, loader)