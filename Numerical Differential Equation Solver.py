# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:11:11 2022

@author: rish3
"""

import numpy as np
import matplotlib.pyplot as plt
#scipy


###########################################################
#Numerical solver for Dc'' = iwc, c'(0) = a, -D*c'(1) = j_hat#
###########################################################

#Number of steps 
#N = int(input("Number of steps: "))+1
N = 100

#Boundary Conditions and constants
j_hat = 1
D = 1
a1 = 0
a2 = -j_hat/D

def get_solutions(ws):
    #gives answers for a list of frequencies
    answers = []
    for w in ws:
        #Uses ghost points
        second_derivative_matrix = np.diag(np.full(N+3, -2)) + np.diag(np.full(N+2, 1), -1) + np.diag(np.full(N+2, 1), 1)
        A = D*second_derivative_matrix - w*1.j*np.identity(N+3)/N**2
    
        #c_1 - c_-1 = 2*a1/N
        #c_N+1 - c_N-1 = 2*a2/N
        A[0][0] = -1
        A[0][1] = 0
        A[0][2] = 1
        A[N+2][N] = -1
        A[N+2][N+1] = 0
        A[N+2][N+2] = 1
    
        b = [2*a1/N] + list(np.zeros(N+1)) + [2*a2/N]    
        c = np.linalg.solve(A, b)
        #print(c)
        ans = c[N+1]/j_hat
        
        answers.append(ans)
    return answers

def exact_sols(ws):
    #get exact answers for the differential equation, takes list of frequencies
    answers = []
    for w in ws:
        y = np.sqrt(w/D*1.j)
        ans = - (1/(D*y))*(np.exp(y) + np.exp(-y))/(np.exp(y)-np.exp(-y))
        answers.append(ans)
        
    return answers

def get_errors(frequencies, answers):
    #gets errors from list of frequencies and corresponding answers
    exact_answers = np.array(exact_sols(frequencies))
    errors = np.abs(np.array(answers) - exact_answers)
    
    percentage_errors = 100*errors/np.abs(exact_answers)
    mean_percentage_error = np.average(percentage_errors)
    return percentage_errors, mean_percentage_error
#solve for range of w

def complex_plot(points):
    #make a plot
    x = [point.real for point in points]
    y = [-point.imag for point in points]
    
    # plot the numbers
    plt.scatter(x, y)
    plt.ylabel('-Imaginary')
    plt.xlabel('Real')
    plt.show()

def errors_plot(frequencies, errors):
    plt.scatter(frequencies, errors)
    plt.ylabel('Errors (%)')
    plt.xlabel('frequency')
    plt.show()
    

frequencies = range(1, 1000)
points = get_solutions(frequencies)
complex_plot(points)
percentage_errors, mean_percentage_error = get_errors(frequencies, points)
errors_plot(frequencies, percentage_errors)

print(mean_percentage_error)