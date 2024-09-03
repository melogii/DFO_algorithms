#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import numpy as np

import math

import pseudoinverse

def gen_simplex_gradient(x, D, p):

    dimension = D.shape

    m, n = dimension

    delta = []

    i = 1

    for i in range(n):

        row = np.array(p(x+D[:,i]) - p(x))

        delta.append(row)

    #delta = np.array(delta)

    gradient = np.dot(pseudoinverse.pseudoinverse(D.T), delta)

    return gradient