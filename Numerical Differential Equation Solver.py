# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:11:11 2022

@author: rish3
"""

import numpy as np
import matplotlib.pyplot as plt
import numerical_methods as nm
import time
#scipy


###########################################################
#Numerical solver for Dc'' = iwc, c'(0) = a, -D*c'(1) = j_hat#
###########################################################

#Boundary Conditions and constants
j_hat = 1
D = 1
a1 = 0
a2 = -j_hat/D

def get_matrix_problem(w, N):
    #returns the matrix A and vector b for the matrix problem we need to solve
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
    return A, b
def get_solutions(ws, method, N):
    #gives answers for a list of frequencies
    answers = []
    start_timer = time.time()
    for w in ws:
        A, b = get_matrix_problem(w, N)
        
        #change the next line for different methods from alternate file
        #c = np.linalg.solve(A, b)
        c = nm.solve(A, b, method)
        #print(c)
        ans = c[N+1]/j_hat
        
        answers.append(ans)
        
    end_timer = time.time()
    time_taken = end_timer - start_timer
    
    return answers, time_taken

def get_full_solution_and_errors(w, method, N):
    #Gives the full vector c for a specific frequency and gives errors
    A, b = get_matrix_problem(w, N)
        
    #change the next line for different methods from alternate file
    #c = np.linalg.solve(A, b)
    c = nm.solve(A, b, method)
    
    #print(c)
    c_exact = full_exact_sol(w, N)
    errors = np.abs(c[1:-1] - c_exact)
    
    percentage_errors = 100*errors/np.abs(c_exact)
    
    return c, percentage_errors

def performance_vs_n(w, start_N, end_N, step_N, method):
    #errors and times list has list of N, list of errors and list of times
    errors_and_times = [[], [], []]
    for n in range(start_N, end_N, step_N):
        answer, timer = get_solutions([w], method, n)
        e, error = get_errors_against_frequencies([w], [answer])
        errors_and_times[0].append(n)
        errors_and_times[1].append(error)
        errors_and_times[2].append(timer)
    return errors_and_times
def exact_sols(ws):
    #get exact answers for the differential equation, takes list of frequencies
    answers = []
    for w in ws:
        y = np.sqrt(w/D*1.j)
        ans = - (1/(D*y))*(np.exp(y) + np.exp(-y))/(np.exp(y)-np.exp(-y))
        answers.append(ans)
        
    return answers

def full_exact_sol(w, N):
    #get entire c vector for differential equation with a specific frequency
    answers = []
    for x in np.linspace(0, 1, N+1):
        y = np.sqrt(w/D*1.j)
        ans = - (1/(D*y))*(np.exp(y*x) + np.exp(-y*x))/(np.exp(y)-np.exp(-y))
        answers.append(ans)
        
    return answers

def get_errors_against_frequencies(frequencies, answers):
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

def plot_errors_against_frequencies(frequencies, errors):
    plt.scatter(frequencies, errors)
    plt.ylabel('Errors (%)')
    plt.xlabel('frequency')
    plt.show()
    
def plot_errors_against_x(errors):

    plt.scatter(np.linspace(0, 1, N+1), errors)
    plt.ylabel('Errors (%)')
    plt.xlabel('x')
    plt.show()

def plot_errors_and_times_against_n(errors_and_times):
    fig, ax = plt.subplots()
    
    ax.plot(errors_and_times[0], errors_and_times[1], color = 'red')
    ax.tick_params(axis = 'y', labelcolor = 'red')
    ax2 = ax.twinx()
    ax2.plot(errors_and_times[0], errors_and_times[2], color = 'green')
    ax2.tick_params(axis = 'y', labelcolor = 'green')

    ax.set_ylabel('error (%)')
    ax.set_xlabel('N')
    ax2.set_ylabel('time (s)')
    plt.show()

#Number of steps 
#N = int(input("Number of steps: "))+1
N = 100

#Choose solution method
method = 'standard'

frequencies = range(1, 1000)
points, timer = get_solutions(frequencies, method, N)
complex_plot(points)
percentage_errors, mean_percentage_error = get_errors_against_frequencies(frequencies, points)
plot_errors_against_frequencies(frequencies, percentage_errors)

points, errors = get_full_solution_and_errors(1000, method, N)
plot_errors_against_x(errors)

errors_and_times = performance_vs_n(1000, 50, 1000, 50, 'standard')
plot_errors_and_times_against_n(errors_and_times)

#print(timer)