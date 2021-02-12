# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:06:38 2020

@author: satya
"""
exp = '60'
path = 'Results/federated_system/exp'+exp+'/upload_schedule.txt'

total_upload = []
with open(path) as f:
    for line in f:
        x = f.readline()
        for a in x:                
            try:
                int(a)
            except:
                pass
            else:
                total_upload.append(int(a))
                
for i in range(8):
    print(total_upload.count(i))
        