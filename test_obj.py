# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:21:44 2020

@author: satya
"""
import multiprocessing as mp

class test:
    
    def __init__(self,name):
        
        self.name = name
        
    def print_name(self):
        
        for i in range(1000):
            print(self.name)
            
        
def fixx(i):
    print('Lmao',i)
    return 0
    
if __name__ == '__main__':

    c1 = test('client 1')
    c2 = test('client 2')
    mp.Process(target=c1.print_name).start()
    mp.Process(target=c2.print_name).start()
    print("Done")