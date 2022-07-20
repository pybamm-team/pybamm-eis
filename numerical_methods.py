# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:10:36 2022

@author: rish3
"""
import numpy as np

def solve(A, b, method, start_point = 'none'):
    #This file will contain different algorithms to solve Ax = b
    if method == 'cg':
        if str(start_point) == 'none':
            start_point = np.zeros(np.shape(b))
        c = conjugate_gradient(A, b, start_point)
    elif method == 'standard':
        c = np.linalg.solve(A, b)
    
    return c

def conjugate_gradient(A, b, start_point):

    xk = np.array(start_point)
    rk = np.array(b) - A@xk
    pk = rk
    
    max_num_iter = 10
    rk1rk1 = np.dot(np.conj(rk), rk)
    tol = 10**-5
    
    for k in range(max_num_iter):
        Apk = A@pk
        rkrk = rk1rk1
        pkApk = np.dot(np.conj(pk), Apk)
        
        alpha_k = rkrk / pkApk
        
        xk = xk + alpha_k * pk
        
        if alpha_k*pk[-1] < tol:
            break
        else:
            rk = rk - alpha_k * Apk
            
            rk1rk1 = np.dot(np.conj(rk), rk)
            
            beta_k = rk1rk1 / rkrk
            
            pk = rk + beta_k * pk
        
    return xk


A = np.array([[2, 10.j], [-10.j, 2]])
b = np.array([1, 5])
c_exact = solve(A, b, 'standard')
c = solve(A, b, 'cg')
import scipy.sparse.linalg
c2 = scipy.sparse.linalg.cg(A, b)


e = c - c_exact 
e2 = c2[0] - c_exact

print(c)
print(c_exact)
print(e)

print()
print(c2)
print(e2)



#CG required positive definite symmetric matrices 
#Make matrix symmetric?
#BiCG
#BiCGSTAB?
