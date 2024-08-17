#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import numpy as np

import math

def f(x):

    return pow(x,2)

def g(x, y):

    return math.sqrt(pow(x,2)+ pow(y,2))

def h(x,y,z):

    return 1/(x*(y+z))


# SIMPLEX GRADIENT

# We need to provide: the function, a point in Rn and the directions

x = [0, 0]

D = np.array([[2, 2], [1, -2]])

def m(x):
    return x[0]*x[0] + 2*x[1] - 3

#print(m(x))

delta = np.array([m(x + D[:,0]) - m(x), m(x + D[:,1]) - m(x)])

#print(delta.T)

def simplex_gradient(x, D, m):

    n = len(x)

    delta = []

    i = 1

    for i in range(n):

        row = [m(x+D[:,i]) - m(x)]

        delta.append(row)
    
    #print(f"delta  = {delta}")

    #print(f"(D.T)^(-1) = {np.linalg.inv(D.T)}")

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
    


B = np.array([[1/3, 2, 0], [2, -6, 0]])

#print(pseudoinverse(B))


def gen_simplex_gradient(x, D, p):

    dimension = D.shape

    m, n = dimension

    delta = []

    i = 1

    for i in range(n):

        row = [p(x+D[:,i]) - p(x)]

        delta.append(row)

    gradient = np.dot(pseudoinverse(D.T), delta)

    return gradient

D = np.array([[0.02, 0.02, -0.02], [0.01, -0.02, -0.01]])

print(pseudoinverse(D.T))