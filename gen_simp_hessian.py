#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import numpy as np

import math

import pseudoinverse, gen_simp_grad

def gen_simplex_hessian(x, C, D, p):

    dimension = D.shape

    m, n = dimension

    delta = []

    i = 1

    g_0 = gen_simp_grad.gen_simplex_gradient(x, D, p)

    for i in range(n):

        row = ((gen_simp_grad.gen_simplex_gradient(x + C[:,i], D, p) - g_0).T).flatten()

        delta.append(row)

    C = np.array(C)

    hessian = np.dot(pseudoinverse.pseudoinverse(C.T), delta)

    return hessian