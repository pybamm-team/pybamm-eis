# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:43:07 2022

@author: rish3
"""

# Tridiagonal solve

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:51:34 2022

@author: rish3
"""


import numpy as np
import matplotlib.pyplot as plt
import time
from numerical_methods import tdge

##############################################################
# Numerical solver for Dc'' = iwc, c'(0) = a, -D*c'(1) = j_hat#
##############################################################

# Boundary Conditions and constants
j_hat = 1
D = 1
a1 = 0
a2 = -j_hat / D


def get_matrix_problem(w, N):
    # returns the matrix A and vector b for the matrix problem we need to solve
    # Uses ghost points
    # altered with EROS to be tridiagonal
    # returns 3 diagonals
    diag2 = np.full(N + 3, -2 - w * 1.0j / N**2)
    diag1 = np.full(N + 2, 1, dtype=float)
    diag3 = np.full(N + 2, 1, dtype=float)

    # c_1 - c_-1 = 2*a1/N
    # c_N+1 - c_N-1 = 2*a2/N
    diag2[0] = -1
    diag2[-1] = 2
    diag1[-1] = -2

    b = [2 * a1 / N] + list(np.zeros(N + 1)) + [2 * a2 / N]
    b = np.array(b, dtype=float)
    return diag1, diag2, diag3, b


def get_solutions(ws, N):
    # gives answers for a list of frequencies
    answers = []
    start_timer = time.time()
    # start_point = initial_guess(ws[0], N)
    for w in ws:
        diag1, diag2, diag3, b = get_matrix_problem(w, N)
        # change the next line for different methods from alternate file
        # c = np.linalg.solve(A, b)
        # c = scipy.sparse.linalg.bicgstab(A, b, x0 = start_point)
        c = tdge(diag1, diag2, diag3, b)
        ans = c / j_hat

        answers.append(ans)
        # start_point = c[0]
    end_timer = time.time()
    time_taken = end_timer - start_timer

    return answers, time_taken


def initial_guess(w, N):
    A, b = get_matrix_problem(w, 100)
    c = np.linalg.solve(A, b)
    c_initial = []
    s = int((N + 3) / 103) + 1
    for i in range(N + 3):
        a = int(i / s)
        c_initial.append(c[a])
    return np.array(c_initial)


def exact_sols(ws):
    # get exact answers for the differential equation, takes list of frequencies
    answers = []
    for w in ws:
        y = np.sqrt(w / D * 1.0j)
        ans = -(1 / (D * y)) * (np.exp(y) + np.exp(-y)) / (np.exp(y) - np.exp(-y))
        answers.append(ans)

    return answers


def get_errors_against_frequencies(frequencies, answers):
    # gets errors from list of frequencies and corresponding answers
    exact_answers = np.array(exact_sols(frequencies))
    errors = np.abs(np.array(answers) - exact_answers)

    percentage_errors = 100 * errors / np.abs(exact_answers)
    return errors, percentage_errors


def complex_plot(points):
    # make a plot
    x = [point.real for point in points]
    y = [-point.imag for point in points]

    # plot the numbers
    plt.scatter(x, y)
    plt.ylabel("-Imaginary")
    plt.xlabel("Real")
    plt.show()


def plot_errors_against_frequencies(frequencies, errors):
    plt.scatter(frequencies, errors)
    plt.ylabel("Errors")
    plt.xlabel("frequency")
    plt.show()


# Number of steps
# N = int(input("Number of steps: "))+1
N = 10000

frequencies = range(1, 1000)
points, timer = get_solutions(frequencies, N)
print(points)
complex_plot(points)
errors, percentage_errors = get_errors_against_frequencies(frequencies, points)
plot_errors_against_frequencies(frequencies, errors)
plot_errors_against_frequencies(frequencies, percentage_errors)
print(timer)
