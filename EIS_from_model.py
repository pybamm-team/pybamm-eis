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

def EIS(model, start_freq, end_freq, num_points, method = 'auto'):
    #model should be a pybamm object
    
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
    
    # gives answers for a list of frequencies
    answers = []
    ws = []
    w = start_freq
    
    if method == 'prebicgstab':
        k = get_k(M)
        M_diags, J_diags = get_block_diagonals(M, J, k)
        
        A_diag = []
        for i, B in enumerate(J_diags[1]):
            A_diag.append(B + 1.j*start_freq*M_diags[i])
        
        L, U = nm.ILUpreconditioner(J_diags[0], A_diag, J_diags[2])
        start_point = np.linalg.solve(L, b)
        start_point = np.linalg.solve(U, start_point)
    else:
        start_point = 'zero'
        
    w_increment_max = (end_freq - start_freq) / num_points

    while w <= end_freq:
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
            
        ans = c[-1][0]
        
        answers.append(ans)
        
        es = np.abs(np.array(stored_vals) - ans)
        ns = num_iters+1 - np.array(ns)
        
        old_c = np.array(c)
        if len(answers) == 1:
            w_increment = float(w_increment_max)
            start_point = c
        else:
            kappa = (np.abs(ans - start_point[-1]))/w_increment**2
            ys = []
            for j, e in enumerate(es):
                y = 2*ns[j]/(-w_increment+np.sqrt(w_increment**2+4*(e+0.001)/kappa))
                ys.append(y)
            y_min = min(ys)
            
            if ys[-1] == y_min:
                try:
                    y_min = max(2*y_min - ys[-2], [0.01])
                except IndexError:
                    pass
                n_val = ns[-1] + 5
                w_increment = min((n_val/y_min)[0], w_increment_max)
            else:                
                w_increment = min((ns[ys.index(y_min)]/y_min)[0], w_increment_max)
                
            old_increment = float(w_increment)
            start_point = c + w_increment/old_increment * (c - old_c)

        


        ws.append(w)
        w = w + w_increment

    return answers, ws

def thomas_method(M, J, b, start_freq, end_freq, num_points, k=0):
    
    if k == 0:
        k = get_k(M)
    
    answers = []
    
    
    ws = np.linspace(start_freq, end_freq, num_points)
        

    if k == 1:
        M_diags, J_diags = get_diagonals(M, J)
    else:
        M_diags, J_diags = get_block_diagonals(M, J, k)
        
        
    for w in ws:
        A_diag = []
        for i, B in enumerate(J_diags[1]):
            A_diag.append(B + 1.j*w*M_diags[i])
        if k == 1:
            ans = nm.thomasMethod(J_diags[0], A_diag, J_diags[2], b)
        else:
            ans = nm.thomasBlockMethod(J_diags[0], A_diag, J_diags[2], b)
        answers.append(ans)
        ws.append(w)
    
    return answers, ws
            
        
def get_k(M):
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
        M_diag.append(J[i, i])
    diag2.append(J[n-1, n-1])
    M_diag.append(J[n-1, n-1])
    J_diags = (diag1, diag2, diag3)
    return M_diag, J_diags

                    
                        
                        
                
            