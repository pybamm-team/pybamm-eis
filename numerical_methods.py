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
    #choose x0 = 0
    xk = np.zeros(np.shape(b))
    rk = np.array(b)
    pk = rk
    
    num_iter = 2
    rk1rk1 = np.dot(rk, rk)
    for k in range(num_iter):
        Apk = A@pk
        rkrk = rk1rk1
        pkApk = np.dot(pk, Apk)
        
        alpha_k = rkrk / pkApk
        xk = xk + alpha_k * pk
        
        rk = rk - alpha_k * Apk
        
        rk1rk1 = np.dot(rk, rk)
        
        beta_k = rk1rk1 / rkrk
        
        pk = rk + beta_k * pk

    return xk

'''
A = np.array([[2, 100], [100, 2]])
b = np.array([1, 5])
c_exact = solve(A, b, 'standard')
c = solve(A, b, 'cg')

e = c - c_exact 

print(c)
print(c_exact)
print(e)
'''
#CG required positive definite symmetric matrices 
#Make matrix symmetric?
#BiCG
#BiCGSTAB?
