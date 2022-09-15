# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:00:27 2022

@author: rish3
"""
import pybamm
import numerical_methods as nm

def EIS(model, start_freq, end_freq, num_points, method = 'auto'):
    #model should be a pybamm object
    
    solver = pybamm.ScipySolver()
    solver.set_up(model)
    
    y0 = model.concatenated_initial_conditions.entries  # vector of initial conditions
    J = model.jac_rhs_algebraic_eval(0, y0, []).sparse()  #  call the Jacobian and return a (sparse) matrix
    
    b = model.rhs_algebraic_eval(0, y0, [])
    M = model.mass_matrix.entries
    
    #A = iwM - J, Ac = b
    
    
    if method == "bicgstab":
        
    elif method == "prebicgstab":
        
    elif method == "cg":
        
    elif method == "thomas":
        
    elif method == "auto":
        
    else:
        raise Exception("Not a valid method")
        
        
def iterative_method(M, J, b, start_freq, end_freq, num_points, method):
    
    # gives answers for a list of frequencies
    answers = []
    start_timer = time.time()
    ws = []
    w = start_freq
    
    if method == 'prebicgstab':
        L, U = nm.ILUpreconditioner()
        start_point = initial_guess(w, N)
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
            
        ans = c[-1]
        
        answers.append(ans)
        
        es = np.abs(np.array(stored_vals) - ans)
        ns = num_iters+1 - np.array(ns)
        
        if len(answers) != 1:
            kappa = (ans - start_point[-1])/w_increment**2
            ys = []
            for j, e in enumerate(es):
                y = 2*ns[j]/(-w_increment+np.sqrt(w_increment**2+4*e/kappa))
                ys.append(y)
            y_min = min(ys)
            
            if ys[-1] = y_min:
                y_min = 2*y_min - ys[-2]
                n_val = ns[-1] + 5
                w_increment = min(n_val/y_min, w_increment_max)
            else:                
                w_increment = min(ns[ys.index(y_min)]/y_min, w_increment_max)
                
            old_increment = w_increment
            start_point = c + w_increment/old_increment * (c - old_c)
        else:
            w_increment = w_increment_max
        
        old_c = c

        ws.append(w)
        w = w + w_increment

    end_timer = time.time()
    time_taken = end_timer - start_timer

    return answers, ws, time_taken  

def get_diagonals(A):
    for i in range(np.shape(A)[0]):
        if A[i, 0] == 0 :