#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import numpy as np

import math

def simplex_gradient(x, D, m):

    n = len(x)

    delta = []

    i = 1

    for i in range(n):

        row = [m(x+D[:,i]) - m(x)]

        delta.append(row)

    gradient = np.dot(np.linalg.inv(D.T), delta)

    return gradient

#print(simplex_gradient(x, D, m))

def pseudoinverse(A):

    # we assume that A has full rank

    dimension = A.shape

    m, n = dimension

    rank = np.linalg.matrix_rank(A)

    #Full rank matrix

    if (m == rank):
         
        if m == n:

            return np.linalg.inv(A)
        
        elif m < n:

            return np.dot(A.T, np.linalg.inv(np.dot(A, A.T)))
        
    else: #(case where n == rank) and m > n

        return np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
    


def gen_simplex_gradient(x, D, p):

    dimension = D.shape

    m, n = dimension

    delta = []

    i = 1

    for i in range(n):

        row = np.array(p(x+D[:,i]) - p(x))

        delta.append(row)

    gradient = np.dot(pseudoinverse(D.T), delta)

    return gradient


def gen_simplex_hessian(x, C, D, p):

    dimension = D.shape

    m, n = dimension

    delta = []

    i = 1

    g_0 = gen_simplex_gradient(x, D, p)

    for i in range(n):

        row = ((gen_simplex_gradient(x + C[:,i], D, p) - g_0).T).flatten()

        delta.append(row)

    C = np.array(C)

    hessian = np.dot(pseudoinverse(C.T), delta)

    return hessian



#  WE ARE MEASURING THE NORM OF THE DIFFERENCE BETWEEN THE GEN SIMPLEX HESSIAN AND THE ACTUAL HESSIAN FOR A COUPLE FUNCTIONS

def p(x):

    return x[0]*x[0]*x[0] + x[1]*x[1]*x[1] + x[2]*x[2]*x[2]
    
def q(x):

    return x[0]*np.exp(x[0] + x[1])

x_p = np.array([2, -1, 1])
    
D_p = np.array([[0.1, 0, 0, -0.1, 0], [0, 0.1, -0.1, -0.2, 0.2], [-0.1, 0, 0.1, 0.2, -0.2]])

x_q = np.array([0,0])

D_q = np.array([[0.1, 0], [0, 0.1]]) 

stop = False

while stop == False:

    for k in range(9):

        delta = pow(10, -k)

        D_p = delta * D_p

        norm_p = np.linalg.norm(gen_simplex_hessian(x_p, D_p, D_p, p) - np.array([[12, 0, 0], [0, -6, 0], [0, 0, 6]]))

        D_q = pow(10, -k) * D_q

        norm_q = np.linalg.norm(gen_simplex_hessian(x_q, D_q, D_q, q) - np.array([[2, 1], [1, 0]]))

        print("delta = ", delta, " accuracy for p: ", norm_p, " accuracy for q ", norm_q)

    stop = True

    


    
    





