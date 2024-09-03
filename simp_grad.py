#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import numpy as np

import math

def m(x):
    return x[0]*x[0] + 2*x[1] - 3

def simplex_gradient(x, D, m):

    n = len(x)

    delta = []

    i = 1

    for i in range(n):

        row = [m(x+D[:,i]) - m(x)]

        delta.append(row)

    gradient = np.dot(np.linalg.inv(D.T), delta)

    return gradient

#x = [0, 0]

#D = np.array([[2, 2], [1, -2]])

#print(simplex_gradient(x, D, m))