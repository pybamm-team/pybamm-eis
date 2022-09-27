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

def EIS(model, start_freq, end_freq, num_points, method = 'auto'):
    '''
    EIS(model, start_freq, end_freq, num_points, method = 'auto')
    
    calculates impedence for a range of frequencies

    Parameters
    ----------
    model : Pybamm model object
        First set this up using Pybamm.
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
        thomas - only use for tridiagonal or block tridiagonal matrices where
        each block is the same size.
        bicgstab - use for matrices where no preconditioner is known
        prebicgstab - use for matrices where a preconditioner is known - for
        example near block tridiagonal matrices.
        
        The default is 'auto'.

    Raises
    ------
    Exception
        If invalid data is entered.

    Returns
    -------
    answers : list
        Complex values of impedence at each frequecy.
    ws : list
        Frequencies evaluated at.
    time_taken : float
        How long the calculation took.

    '''
    
    solver = pybamm.ScipySolver()
    solver.set_up(model)
    
    y0 = model.concatenated_initial_conditions.entries  # vector of initial conditions
    J = model.jac_rhs_algebraic_eval(0, y0, []).sparse()  #  call the Jacobian and return a (sparse) matrix
    
    b = model.rhs_algebraic_eval(0, y0, [])
    M = model.mass_matrix.entries

    #A = iwM - J, Ac = b
    
    start_timer = time.time()
    
    if method == "bicgstab" or method == "prebicgstab" or method == "cg":
        answers, ws = iterative_method(M, J, b, start_freq, end_freq, num_points, method)    
    elif method == "thomas":
        answers, ws = thomas_method(M, J, b, start_freq, end_freq, num_points)
    elif method == "auto":
        k = get_k(M)
        if k >= 8:
            answers, ws = iterative_method(M, J, b, start_freq, end_freq, num_points, 'bicgstab')
        else:
            answers, ws = thomas_method(M, J, b, start_freq, end_freq, num_points, k=k)
    else:
        raise Exception("Not a valid method")
    
    end_timer = time.time()
    time_taken = end_timer - start_timer
    
    return answers, ws, time_taken
def iterative_method(M, J, b, start_freq, end_freq, num_points, method):
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

    Returns
    -------
    answers : list
        Complex values of impedence at each frequecy.
    ws : list
        Frequencies evaluated at.

    '''

    # gives answers for a list of frequencies
    answers = []
    ws = []
    w = start_freq
    
    if method == 'prebicgstab':
        k = get_k(M)
        M_diags, J_diags = get_block_diagonals(M, J, k)
        
        A_diag = []
        for i, B in enumerate(J_diags[1]):
            A_diag.append(1.j*start_freq*M_diags[i]-B)
        
        L, U = nm.ILUpreconditioner(J_diags[0], A_diag, J_diags[2])
        #L = scipy.sparse.eye(np.shape(b)[0])
        #U = scipy.sparse.eye(np.shape(b)[0])
        start_point = scipy.sparse.linalg.spsolve(L, b)
        start_point = np.array(scipy.sparse.linalg.spsolve(U, start_point))
        start_point = np.reshape(start_point, np.shape(b))
    else:
        start_point = 'zero'
        
    w_log_increment_max = (np.log(end_freq) - np.log(start_freq)) / num_points
    multiplier = np.exp(w_log_increment_max) - 1
    while w <= end_freq:
        #print(w)
        w_increment_max = w*multiplier
        A = 1.j*w*M - J
        num_iters = 0
        stored_vals = []
        ns = []
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
        ans = c[-1][0]
        
        answers.append(ans)
        
        es = np.abs(np.array(stored_vals) - ans)
            
        ns = num_iters+1 - np.array(ns)
        
        old_c = np.array(c)
        if len(answers) == 1:
            w_increment = float(w_increment_max)
            start_point = c
        else:
            old_increment = float(w_increment)
            kappa = np.abs(ans - start_point[-1])/(w_increment/end_freq)**2
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
                w_increment = min((end_freq*n_val/y_min)[0], w_increment_max)
            else:                
                w_increment = min((end_freq*ns[ys.index(y_min)]/y_min)[0], w_increment_max)
                
            #print(y_min)
            start_point = c + w_increment/old_increment * (c - old_c)

        #if w_increment == w_increment_max:
        #    print("MAX")


        ws.append(w)
        w = w + w_increment

    return answers, ws

def thomas_method(M, J, b, start_freq, end_freq, num_points, k=0):
    '''
    thomas_method(M, J, b, start_freq, end_freq, num_points, k=0)
    
    calculates impedence for a range of frequencies using the Thomas method

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
    k : int, optional
        the size of the blocks in the matrices. If 0 entered this is calculated
        automatically.

    Returns
    -------
    answers : list
        Complex values of impedence at each frequecy.
    ws : list
        Frequencies evaluated at.

    '''
    if k == 0:
        k = get_k(M)
    
    answers = []
    
    
    ws = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), num_points))
    '''
    if k == 1:
        M_diags, J_diags = get_diagonals(M, J)
    else:
        M_diags, J_diags = get_block_diagonals(M, J, k)   
    for w in ws:
        A_diag = []
        for i, B in enumerate(J_diags[1]):
            A_diag.append(1.j*w*M_diags[i]-B)
        if k == 1:
            ans = nm.thomasMethod(J_diags[0], A_diag, J_diags[2], b)
        else:
            ans = nm.thomasBlockMethod(J_diags[0], A_diag, J_diags[2], b)
        answers.append(ans)
    '''
    for w in ws:
        A = 1.j*w*M - J
        ans = scipy.sparse.linalg.spsolve(A, b)[-1]
        answers.append(ans)
    
    return answers, ws
            
        
def get_k(M):
    '''
    get_k(M)
    
    gets the block size from a block diagonal matrix
    
    Parameters
    ----------
    
    M : scipy sparse csr matrix 
        Must be block diagonal.
    
    Returns
    -------
    
    k : int
        the block size
    '''

    n = np.shape(M)[0]
    for i in range(np.shape(M)[0]):
        if M[i, 0] == 0:
            if all(M[i, j]==0 for j in range(i)):
                counter = 0
                for m in range(i):
                    if all((M[m, j]==0 and M[j, m]==0) for j in range(i, n)):
                        counter += 1
                    else:
                        break
                if counter == i:
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
        diag1.append(scipy.sparse.csr_matrix.todense(J[range(i*k, (i+1)*k), :][:, range((i+1)*k, (i+2)*k)]).astype('float64'))
        diag2.append(scipy.sparse.csr_matrix.todense(J[range(i*k, (i+1)*k), :][:, range((i)*k, (i+1)*k)]).astype('float64'))
        diag3.append(scipy.sparse.csr_matrix.todense(J[range((i+1)*k, (i+2)*k), :][:, range((i)*k, (i+1)*k)]).astype('float64'))
        M_diag.append(scipy.sparse.csr_matrix.todense(M[range(i*k, (i+1)*k), :][:, range((i)*k, (i+1)*k)]).astype('float64'))
    diag2.append(scipy.sparse.csr_matrix.todense(J[range((m-1)*k, m*k), :][:, range((m-1)*k, (m)*k)]).astype('float64'))
    M_diag.append(scipy.sparse.csr_matrix.todense(M[range((m-1)*k, m*k), :][:, range((m-1)*k, (m)*k)]).astype('float64'))
    return M_diag, (diag1, diag2, diag3)

def get_diagonals(M, J):
    '''
    get_diagonals(M, J)
    
    converts a scipy sparse csr matrix to a diagonal storage form
    
    Parameters
    ----------
    
    M : scipy sparse csr matrix 
        Must be diagonal
    J : scipy sparse csr matrix 
        Must be tridiagonal
    
    Returns
    -------
    
    M_diag : list
        list of all the entries (floats) on the diagonal of M
        
    J_diags = (diag1, diag2, diag3) : tuple
        a tuple of lists of entries (floats) on the 3
        diagonals of J. 1 is below the main diagonal, 2 is the 
        main diagonal, 3 is above.
    '''
    #for k = 1 only
    n = np.shape(M)[0]
    diag1 = []
    diag2 = []
    diag3 = []
    M_diag = []
    for i in range(n-1):
        diag1.append(J[i, i+1])
        diag2.append(J[i, i])
        diag3.append(J[i+1, i])
        M_diag.append(M[i, i])
    diag2.append(J[n-1, n-1])
    M_diag.append(M[n-1, n-1])
    J_diags = (diag1, diag2, diag3)
    return M_diag, J_diags

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