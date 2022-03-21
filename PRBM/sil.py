# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:10:00 2022

@author: Asus
"""
import numpy as np
nrb = 4
iteration = 0
Ti = np.ones([4,4])
theta = np.array([1,1,1])
gamma = np.array([[0.12512513, 0.35035035, 0.38838839, 0.13613614]])
l = np.array([[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]])

for k in range(0,nrb-1):
    temp = np.array([np.cos(theta[k]),0,np.sin(theta[k]),
              gamma[0][k+1]*l[0][iteration]*np.sin(theta[k]),
              0,1,0,0,
              -1*np.sin(theta[k]),
              0,np.cos(theta[k]),gamma[0][k+1]*l[0][iteration]*np.cos(theta[k]),
              0, 0, 0, 1]).reshape(4,4);
    Ti = np.matmul(Ti,temp);