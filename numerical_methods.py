# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:10:36 2022

@author: rish3
"""
import numpy as np
import scipy.sparse

def empty(xk):
    #An empty callback function. Callbacks can be written as desired.
    pass

def conjugate_gradient(A, b, start_point = 'zero', callback = empty, tol = 10**-5):
    #Uses conjugate gradient algorithm
    #Should not be used unless A Hermitian. If A not hermitian, use BicgSTAB instead.
    #For best performance A should be a scipy csr sparse matrix
    if str(start_point) == 'zero':
        start_point = np.zeros(np.shape(b))
    
    xk = np.array(start_point)
    rk = np.array(b) - A@xk
    pk = rk
    
    max_num_iter = len(b)
    rk1rk1 = np.dot(np.conj(rk), rk)
    
    
    for k in range(max_num_iter):
        Apk = A@pk
        rkrk = rk1rk1
        pkApk = np.dot(np.conj(pk), Apk)
        
        alpha_k = rkrk / pkApk
        
        xk = xk + alpha_k * pk
        
        callback(xk)
        
        #Stop if the change in the last entry is under tolerance
        if alpha_k*pk[-1] < tol:
            break
        else:
            rk = rk - alpha_k * Apk
            
            rk1rk1 = np.dot(np.conj(rk), rk)
            
            beta_k = rk1rk1 / rkrk
            
            pk = rk + beta_k * pk
        
    return xk


def bicgstab(A, b, start_point = 'zero', callback = empty, tol = 10**-5):
    #Uses BicgSTAB algorithm with no preconditioner
    #For best performance A should be a scipy csr sparse matrix
    
    if str(start_point) == 'zero':
        start_point = np.zeros(np.shape(b))
        
    xk = np.array(start_point)
    rk = np.array(b) - A@xk
    r0 = np.conj(rk)
    rhok = 1
    alpha_k = 1
    wk = 1
    
    pk = np.zeros(np.shape(b))
    vk = pk
    max_num_iter = 2*np.shape(b)[0]
    
    for k in range(1, max_num_iter+1):
        rhok1 = rhok
        rhok = np.dot(np.conj(r0.T), rk)
        beta_k = (rhok/rhok1)*(alpha_k/wk)
        
        pk = rk + beta_k*(pk - wk*vk)
        vk = A@pk
        
        alpha_k = rhok/np.dot(np.conj(r0.T), vk)
        h = xk + alpha_k*pk
        
        s = rk - alpha_k*vk
        t = A@s
        
        wk = np.dot(np.conj(t.T), s)/np.dot(np.conj(t.T), t)
        xk1 = xk
        xk = h + wk*s
        callback(xk)
        if np.abs(xk[-1] - xk1[-1]) < tol:
            break
        else:
            rk = s - wk*t  
    return xk

def prebicgstab(A, b, L, U, start_point = 'zero', callback = empty):
    #Uses preconditioned bicgstab
    #the preconditioner is in the form K = LU where K is approximately A
    #For best performance A should be a scipy csr sparse matrix
    
    if str(start_point) == 'zero':
        start_point = np.zeros(np.shape(b))
        
    xk = np.array(start_point)
    rk = np.array(b) - A@xk
    r0 = np.conj(rk)
    rhok = 1
    alpha_k = 1
    wk = 1
    
    pk = np.zeros(np.shape(b))
    vk = pk
    
    max_num_iter = 2*np.shape(b)[0]
    tol = 10**-5
    
    for k in range(1, max_num_iter+1):
        rhok1 = rhok
        rhok = np.dot(np.conj(r0.T), rk)
        beta_k = (rhok/rhok1)*(alpha_k/wk)
        
        pk = rk + beta_k*(pk - wk*vk)
        
        y = np.linalg.solve(L, pk)
        y = np.linalg.solve(U, y)
        
        vk = A@y
        
        alpha_k = rhok/np.dot(np.conj(r0.T), vk)
        
        h = xk + alpha_k * y

        s = rk - alpha_k*vk
        
        z = np.linalg.solve(L, s)
        z = np.linalg.solve(U, z)
        
        t = A@z
        
        wk = np.dot(np.conj(t.T), s)/np.dot(np.conj(t.T), t)
        
        xk1 = xk
        xk = h + wk*z
        callback(xk)
        if np.abs(xk[-1] - xk1[-1]) < tol:
            break
        else:
            rk = s - wk*t  
    return xk

def thomasMethod(diag1, diag2, diag3, b):
    #For tridiagonal matrices
    #first convert the matrix into lists of the diagonals
    #diag2 and b must have 1 more entry than 1 and 3
    #diag2 cannot have any zeros
    #b is zeros followed by an entry
    
    n = len(diag1)
    for i in range(n):
        diag2[i+1] -= diag3[i]*diag1[i]/diag2[i]
    ans = complex(b[n])/diag2[n]
    return ans

def thomasBlockMethod(diag1, diag2, diag3, b):
    #For block tridiagonal matrices
    #first convert the matrix into lists of the diagonals (lists of matrices)
    #similar to previous but lists contain square matrices of equal size
    #diag2 and b must have 1 more entry than 1 and 3
    #diag2 cannot have any non-invertible matrices
    #b is zeros followed by k entries, k is block size
    m = len(diag1)
    for i in range(m):
        Q = np.linalg.solve(diag2[i], diag3[i])
        diag2[i+1] -= diag1[i]@Q
    
    Bm = diag2[m]
    k = np.shape(Bm)[0]
    y = b[-k:]
    
    ans = np.linalg.solve(Bm, y)[-1]
    
    return ans

def ILUpreconditioner(diag1, diag2, diag3):
    #For block tridiagonal matrices
    #first convert the matrix into lists of the diagonals (lists of matrices)
    #diag2 and b must have 1 more entry than 1 and 3
    #diag2 cannot have any non-invertible matrices
    m = len(diag2)
    
    k = np.shape(diag1[0])[0]
    Si = diag2[0]
    Ss = [Si]
    Ti = np.linalg.solve(Si, diag3[0])
    Ts = [Ti]
    
    prev = []
    
    for i in range(2, m+1):
        current = [diag2[i-1], diag1[i-2], diag2[i-2], diag3[i-3], diag3[i-2]]
        if current == prev:
            Ss.append(Si)
            Ts.append(Ti)
        else:
            M = np.linalg.solve(current[2], current[3])
            H = current[1]@M
            G = np.linalg.solve(current[0], H)
            F = np.eye(k) + 2*G
            Si = current[0]@np.linalg.inv(F)
            Ss.append(Si)
            
            if i != m:
                Ti = np.linalg.solve(Si, current[4])
                Ts.append(Ti)
                prev = current

    L = scipy.sparse.bmat([[Ss[i] if i == j else diag1[j] if i-j==1
                else None for i in range(m)]
                for j in range(m)], format='csr').A
    U = scipy.sparse.bmat([[np.eye(k) if i == j else Ts[i] if i-j==-1
            else None for i in range(m)]
            for j in range(m)], format='csr').A
    
    return L, U