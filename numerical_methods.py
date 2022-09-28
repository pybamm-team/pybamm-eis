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
    '''
    conjugate_gradient(A, b, start_point = 'zero', callback = empty, tol = 10**-5)
    
    Uses the conjugate gradient method to solve Ax = b
    
    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy 1xn array
    start_point : numpy 1xn array, optional
        Where the iteration starts. The default is 'zero'.
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 10**-5.

    Returns
    -------
    xk : numpy 1xn array
        The solution to Ax = b.

    '''
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
    '''
    bicgstab(A, b, start_point = 'zero', callback = empty, tol = 10**-5)
    
    Uses the BicgSTAB method to solve Ax = b
    
    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy 1xn array
    start_point : numpy 1xn array, optional
        Where the iteration starts. The default is 'zero'.
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 10**-5.

    Returns
    -------
    xk : numpy 1xn array
        The solution to Ax = b.

    '''
    
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
        rhok = np.dot(r0.T, rk)
        beta_k = (rhok/rhok1)*(alpha_k/wk)
        
        pk = rk + beta_k*(pk - wk*vk)
        vk = A@pk
        
        alpha_k = rhok/np.dot(r0.T, vk)
        
        h = xk + alpha_k*pk
        
        if np.linalg.norm(xk - h, 1) < tol:
            xk = h
            callback(xk)
            break
        
        s = rk - alpha_k*vk
        t = A@s
        
        wk = np.dot(np.conj(t.T), s)/np.dot(np.conj(t.T), t)
        xk1 = xk
        xk = h + wk*s
        callback(xk)
        if np.linalg.norm(xk - xk1, 1) < tol:
            break
        else:
            rk = s - wk*t  
    return xk

def prebicgstab(A, b, L, U, start_point = 'zero', callback = empty, tol = 10**-5):
    '''
    prebicgstab(A, b, L, U, start_point = 'zero', callback = empty)
    
    Uses the preconditioned BicgSTAB method to solve Ax = b. The preconditioner
    is of the form LU.
    
    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy 1xn array
    L : scipy sparse csr matrix
        A lower (block) triangular matrix
    U : scipy sparse crs matrix
        An Upper (block) triangular matrix
    start_point : numpy 1xn array, optional
        Where the iteration starts. The default is 'zero'.
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 10**-5.

    Returns
    -------
    xk : numpy 1xn array
        The solution to Ax = b.

    '''
    size = np.shape(b)
    
    if str(start_point) == 'zero':
        start_point = np.zeros(size)
        
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
        rhok = np.dot(r0.T, rk)
        beta_k = (rhok/rhok1)*(alpha_k/wk)
        pk = rk + beta_k*(pk - wk*vk)
        
        y = scipy.sparse.linalg.spsolve(L, pk)
        y = np.reshape(np.array(scipy.sparse.linalg.spsolve(U, y)), size)
        
        vk = A@y
        alpha_k = rhok/np.dot(r0.T, vk)

        h = xk + alpha_k * y
        
        if np.linalg.norm(xk - h, 1) < tol:
            xk = h
            callback(xk)
            break
        
        s = rk - alpha_k*vk

        z = scipy.sparse.linalg.spsolve(L, s)
        z = np.resize(np.array(scipy.sparse.linalg.spsolve(U, z)), size)
        
        t = A@z
        
        wk = np.dot(np.conj(t.T), s)/np.dot(np.conj(t.T), t)
        xk1 = xk
        xk = h + wk*z
        callback(xk)
        if np.linalg.norm(xk - xk1, 1) < tol:
            break
        else:
            rk = s - wk*t  
    return xk

def thomasMethod(diag1, diag2, diag3, b):
    '''
    thomasMethod(diag1, diag2, diag3, b)
    
    Uses the Thomas method to solve Ax = b. A is tridiagonal.
    
    Parameters
    ----------
    diag1 : list
        The lower diagonal of A.
    diag2 : list
        The main diagonal of A
    diag3 : list
        The upper diagonal of A
    b : numpy 1xn array
        Has only one non-zero entry and that is the last entry.

    Returns
    -------
    ans : float
        The last entry of the solution to Ax = b.

    '''
    
    n = len(diag1)
    for i in range(n):
        diag2[i+1] -= diag3[i]*diag1[i]/diag2[i]
    ans = complex(b[n])/diag2[n]
    return ans

def thomasBlockMethod(diag1, diag2, diag3, b):
    '''
    thomasBlockMethod(diag1, diag2, diag3, b)
    
    Uses the Thomas method to solve Ax = b. A is block tridiagonal.
    
    Parameters
    ----------
    diag1 : list of matrices
        The lower diagonal of A.
    diag2 : list of matrices
        The main diagonal of A
    diag3 : list of matrices
        The upper diagonal of A
    b : numpy 1xn array
        Has only non-zero entries alligning with the last block.

    Returns
    -------
    ans : float
        The last entry of the solution to Ax = b.

    '''
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
    '''
    ILUpreconditioner(diag1, diag2, diag3)
    
    Creates an ILU preconditioner for A. A is block tridiagonal.
    
    Parameters
    ----------
    diag1 : list of matrices
        The lower diagonal of A.
    diag2 : list of matrices
        The main diagonal of A
    diag3 : list of matrices
        The upper diagonal of A

    Returns
    -------
    L : scipy sparse csr matrix
        A lower triangular matrix 
    U : scipy sparse csr matrix
        An upper triangular matrix 
        
        We have A equals approximately LU

    '''
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
                else None for j in range(m)]
                for i in range(m)], format='csr')
    U = scipy.sparse.bmat([[np.eye(k) if i == j else Ts[i] if i-j==-1
            else None for j in range(m)]
            for i in range(m)], format='csr')
    
    return L, U