# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:24:56 2022

@author: rish3
"""
##################
###Timings Test###
##################

import numpy as np
from numerical_methods import solve
import scipy.sparse.linalg
from scipy.sparse import csr_matrix
import time

N = 10000  # This takes about 20 minutes to run at 10000.
D = 1
w = 1
second_derivative_matrix = (
    np.diag(np.full(N + 3, -2))
    + np.diag(np.full(N + 2, 1), -1)
    + np.diag(np.full(N + 2, 1), 1)
)
A = D * second_derivative_matrix - w * 1.0j * np.identity(N + 3) / N**2

# c_1 - c_-1 = 2*a1/N
# c_N+1 - c_N-1 = 2*a2/N
A[0][0] = -1
A[0][1] = 0
A[0][2] = 1
A[N + 2][N] = -1
A[N + 2][N + 1] = 0
A[N + 2][N + 2] = 1

b = [0] + list(np.zeros(N + 1)) + [-1 / N]
print(A)
print(b)

t = time.time()
c_exact = solve(A, b, "standard")
t2 = time.time()
c = solve(A, b, "bicgstab")
# c = scipy.sparse.linalg.cg(A, b)
t3 = time.time()

A = csr_matrix(A)
t4 = time.time()
c_bicg = scipy.sparse.linalg.bicgstab(A, b)
t5 = time.time()


e1 = c[0] - c_exact
e2 = c_bicg[0] - c_exact

print("answers")
print(c)
print(c_exact)
print(c_bicg)
print("errors")
print(e1)
print(e2)

print()
print("times")
s = t2 - t  # Approx 1 minute
s2 = t3 - t2  # Approx 20 mins?
s3 = t5 - t4  # 2 seconds
print(s)
print(s2)
print(s3)
