# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:00:27 2022

@author: rish3
"""
import pybamm
import numerical_methods as nm
import numpy as np
import time
import scipy.sparse
import matplotlib.pyplot as plt

def EIS(M, J, b, start_freq, end_freq, num_points, method = 'auto'):
    #redo this
    '''
    EIS(M, J, b, start_freq, end_freq, num_points, method = 'auto')
    
    calculates impedence for a range of frequencies
    
    solves iwMc = Jc + b
    
    Voltage must be the 2nd to last entry in the matrix and Current should be last
    
    Parameters
    ----------
    M : sparse csr matrix
        The mass matrix
    J : sparse csr matrix
        The Jacobian
    b : numpy 1xn array
        The RHS
    start_freq : float
        The initial frequency in a frequency range to be evaluated at.
    end_freq : float
        The final frequency in a frequency range to be evaluated at.
    num_points : int
        the minimum number of frequencies impedence is calculated at over
        the range.
    method : string, optional
        the numerical algorithm to use. Options are:
        cg - conjugate gradient - only use for Hermitian matrices
        bicgstab - use bicg with no preconditioning
        prebicgstab - use bicg with preconditioning
        direct - use Gaussian elimination
        auto - chooses what is expected to be best automatically
        
        The default is 'auto'.

    Raises
    ------
    Exception
        If invalid data is entered.

    Returns
    -------
    Zs : list
        Complex values of impedence at each frequecy.
    ws : list
        Frequencies evaluated at.
    time_taken : float
        How long the calculation took.

    '''

    #A = iwM - J, Ac = b
    
    start_timer = time.time()
    
    if method == "bicgstab" or method == "prebicgstab" or method == "cg":
        Zs, ws = iterative_method(M, J, b, start_freq, end_freq, num_points, method)    
    elif method == "direct":
        Zs, ws = direct_method(M, J, b, start_freq, end_freq, num_points)
    elif method == "auto":
        k = get_k(M)
        if k >= 8:
            Zs, ws = iterative_method(M, J, b, start_freq, end_freq, num_points, 'bicgstab')
        else:
            Zs, ws = direct_method(M, J, b, start_freq, end_freq, num_points, k=k)
    else:
        raise Exception("Not a valid method")
    
    end_timer = time.time()
    time_taken = end_timer - start_timer
    
    return Zs, ws, time_taken



def ILU(A, M, J, L, U):
    if type(L) == str:
        k = get_k(M)
        M_diags, A_diags = get_block_diagonals(M, A, k)
        
        L, U = nm.ILUpreconditioner(A_diags[0], A_diags[1], A_diags[2])
    return L, U

def G_S(A, M, J, L, U):
    L = scipy.sparse.tril(A, format = 'csr')
    U = 'none'
    return L, U

def G_S_V(A, M, J, L, U):
    #Note L and U are reversed here because this is actually a UL factorisation.
    #This doesn't affect bicgstab.
    
    U = scipy.sparse.tril(A, format = 'csr')
    L = scipy.sparse.triu(A, k = 1, format = 'csr')
    Id = scipy.sparse.eye(A.shape[0], dtype = 'complex', format = 'csr')
    L = L + Id
    return L, U
def iterative_method(M, J, b, start_freq, end_freq, num_points, method, preconditioner = ILU):
    '''
    iterative_method(M, J, b, start_freq, end_freq, num_points, method)
    
    calculates impedence for a range of frequencies using an iterative method

    solves iwMc = Jc + b
    
    Parameters
    ----------
    M : sparse csr matrix
    J : sparse csr matrix
    b : numpy 1xn array
    start_freq : float
        The initial frequency in a frequency range to be evaluated at.
    end_freq : float
        The final frequency in a frequency range to be evaluated at.
    num_points : int
        the minimum number of frequencies impedence is calculated at over
        the range.
    method : string, optional
        the numerical algorithm to use. Options are:
        cg - conjugate gradient - only use for Hermitian matrices
        bicgstab - use for matrices where no preconditioner is known
        prebicgstab - use for matrices where a preconditioner is known - for
        example near block tridiagonal matrices.
    preconditioner: function, optional
        A function that calculates a preconditioner from A, M, J and the previous 
        preconditioner. Returns L, U, triangular. Only relevent when using prebicgstab.
        Default is Gauss-Seidel. Return 'none' as a string for U if only L is
        being used. Return triangular as True if both are triangular and False
        if not (eg block preconditioner).
        
    Returns
    -------
    Zs : list
        Complex values of impedence at each frequecy.
    ws : list
        Frequencies evaluated at.

    '''

    # gives answers for a list of frequencies
    Zs = []
    ws = []
    w = start_freq
    
    L = 'none'
    U = 'none'
        
    start_point = b
        
    w_log_increment_max = (np.log(end_freq) - np.log(start_freq)) / num_points
    multiplier = np.exp(w_log_increment_max) - 1
    iters = []
    while w <= end_freq:
        #print(w)
        w_increment_max = w*multiplier
        A = 1.j*w*M - J
        num_iters = 0
        stored_vals = []
        ns = []
        
        if method == 'prebicgstab':
            L, U = preconditioner(A, M, J, L, U)

        def callback(xk):
            nonlocal num_iters
            num_iters += 1
            if num_iters % 5 == 1:
                stored_vals.append(xk[-1])
                ns.append(num_iters)
        
        if method == 'bicgstab':
            c = nm.bicgstab(A, b, start_point=start_point, callback=callback)
        elif method == 'prebicgstab':
            c = nm.prebicgstab(A, b, L, U, start_point=start_point, callback=callback)
        else:
            c = nm.conjugate_gradient(A, b, start_point=start_point, callback=callback)
        #print(num_iters)

        V = c[-2]
        I = c[-1]
        Z = V/I
        Zs.append(Z)
        
        es = np.abs(np.array(stored_vals) - V)
            
        ns = num_iters+1 - np.array(ns)
        
        old_c = np.array(c)
        if len(Zs) == 1:
            w_increment = float(w_increment_max)
            start_point = c
        else:
            old_increment = float(w_increment)
            kappa = np.abs(V - start_point[-2])/(w_increment/end_freq)**2
            ys = []
            for j, e in enumerate(es):
                y = 2*ns[j]/(-w_increment/end_freq+np.sqrt((w_increment/end_freq)**2+4*(e+0.1)/kappa))
                ys.append(y)
            y_min = min(ys)
            #print(ys)
            #print(ns)
            #plt.scatter(ns, es)
            #plt.show()
            #plt.scatter(ns, ys)
            #plt.show()
            if ys[-1] == y_min:
                n_val = ns[-1]+5
                w_increment = min(end_freq*n_val/y_min[0], w_increment_max)
            else:              
                w_increment = min(end_freq*ns[ys.index(y_min)]/y_min[0], w_increment_max)
                
            #print(y_min)
            start_point = c + w_increment/old_increment * (c - old_c)
            
        #if w_increment == w_increment_max:
        #    print("MAX")


        ws.append(w)
        iters.append(num_iters)
        w = w + w_increment

    plt.plot(ws, iters)
    plt.show()
    return Zs, ws

def direct_method(M, J, b, start_freq, end_freq, num_points):
    '''
    thomas_method(M, J, b, start_freq, end_freq, num_points, k=0)
    
    calculates impedence for a range of frequencies using scipy

    solves iwMc = Jc + b
    
    Parameters
    ----------
    M : sparse csr matrix
        Must be block diagonal
    J : sparse csr matrix
        Must be block tridiagonal
    b : numpy 1xn array
    start_freq : float
        The initial frequency in a frequency range to be evaluated at.
    end_freq : float
        The final frequency in a frequency range to be evaluated at.
    num_points : int
        the minimum number of frequencies impedence is calculated at over
        the range.

    Returns
    -------
    Zs : list
        Complex values of impedence at each frequecy.
    ws : list
        Frequencies evaluated at.

    '''
    Zs = []
    ws = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), num_points))

    for w in ws:
        A = 1.j*w*M - J
        ans = scipy.sparse.linalg.spsolve(A, b)
        V = ans[-2]
        I = ans[-1]
        Z = V/I
        Zs.append(Z)
    
    return Zs, ws
            
        
def get_k(M):
    '''
    get_k(M)
    
    gets the block size from a block diagonal matrix
    
    Parameters
    ----------
    
    M : scipy sparse csr matrix 
        Must be block diagonal. (Or near if using for approximation)
    
    Returns
    -------
    
    k : int
        the block size
    '''

    n = np.shape(M)[0]
    for i in range(np.shape(M)[0]):
        if M[i, 0] == 0:
            if all(M[i, j]==0 for j in range(i)):
                '''
                counter = 0
                for m in range(i):
                    if all((M[m, j]==0 and M[j, m]==0) for j in range(i, n)):
                        counter += 1
                    else:
                        break
                if counter == i:
                '''
                k = int(i)
                break
    return k
def get_block_diagonals(M, J, k):
    '''
    get_block_diagonals(M, J, k)
    
    converts a scipy sparse csr matrix to a block diagonal storage form
    
    Parameters
    ----------
    
    M : scipy sparse csr matrix 
        Must be block diagonal
    J : scipy sparse csr matrix 
        Must be block tridiagonal
    k : int
        is the block size
    
    Returns
    -------
    
    M_diag : list
        list of all the block matrices on the diagonal of M
        
    (diag1, diag2, diag3) : tuple
        a tuple of lists of block matrices on the 3
        diagonals of J. 1 is below the main diagonal, 2 is the main diagonal, 3
        is above.
    '''
    n = np.shape(M)[0]
    m = int(n/k)
    diag1 = []
    diag2 = []
    diag3 = []
    M_diag = []
    for i in range(m-1):
        diag1.append(scipy.sparse.csr_matrix.todense(J[range(i*k, (i+1)*k), :][:, range((i+1)*k, (i+2)*k)]).astype('complex'))
        diag2.append(scipy.sparse.csr_matrix.todense(J[range(i*k, (i+1)*k), :][:, range((i)*k, (i+1)*k)]).astype('complex'))
        diag3.append(scipy.sparse.csr_matrix.todense(J[range((i+1)*k, (i+2)*k), :][:, range((i)*k, (i+1)*k)]).astype('complex'))
        M_diag.append(scipy.sparse.csr_matrix.todense(M[range(i*k, (i+1)*k), :][:, range((i)*k, (i+1)*k)]).astype('complex'))
    diag2.append(scipy.sparse.csr_matrix.todense(J[range((m-1)*k, m*k), :][:, range((m-1)*k, (m)*k)]).astype('complex'))
    M_diag.append(scipy.sparse.csr_matrix.todense(M[range((m-1)*k, m*k), :][:, range((m-1)*k, (m)*k)]).astype('complex'))
    return M_diag, (diag1, diag2, diag3)

def nyquist_plot(points):
    '''
    nyquist_plot(points)
    
    makes a nyquist plot from EIS data
    
    Parameters
    ----------
    
    points: list
        list of complex numbers to be plotted
    '''
    # make a plot
    x = [point.real for point in points]
    y = [-point.imag for point in points]

    # plot the numbers
    plt.scatter(x, y)
    plt.ylabel("-Imaginary")
    plt.xlabel("Real")
    plt.show()