# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:47:31 2020

@author: satya
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataloader import Dataloader
from client import Client
from server import Server
from head_network import Head_Network
from neck_network import Neck_Network

from config import config as cfg

loader = Dataloader(num_classes = 3)    
loader.set_path('Data/by_class')
classes = loader.get_classes()

neck_network        = Neck_Network(cfg['neck_architecture'],load_wt=False)
head_network        = Head_Network(cfg['head_architecture'],load_wt=False)

server = Server('server',neck_network)

neck_network        = Neck_Network(cfg['neck_architecture'],load_wt=False)
head_network        = Head_Network(cfg['head_architecture'],load_wt=False)

client = Client('client',neck_network,head_network)
client.set_global_repo(server.parameter_repo)
client.load_dataset(loader)

for i in range(50):
    
    client.download_params()
    client.forward_pass()
    client.backward_pass()
    client.upload_params()
    
    if(i%10==0):
        server.aggregate_params()
        updated_model = server.update_model(10)
        server.distribute_model(updated_model)