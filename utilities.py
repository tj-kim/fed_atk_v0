# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:00:22 2020

@author: satya
"""

def freeze_layers(network,layers):
    
    for idx in layers:
        network.network[idx].weight.requires_grad = False
        network.network[idx].bias.requires_grad = False
    return network