#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import numpy as np

import math

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

#B = np.array([[1/3, 2, 0], [2, -6, 0]])

#print(pseudoinverse(B))