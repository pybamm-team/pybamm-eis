# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:10:36 2022

@author: rish3
"""
import numpy as np

def solve(A, b, method):
    #This file will contain different algorithms to solve Ax = b
    if method == 'cg':
        c = conjugate_gradient(A, b)
    elif method == 'standard':
        c = np.linalg.solve(A, b)
    
    return c

def conjugate_gradient(A, b):
    #TO DO
    c = 0
    return c