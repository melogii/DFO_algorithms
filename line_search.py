#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import numpy as np

import math

def line_search(f, x, g, eta, itmax = 6):

    t = 1

    k = 1

    norm = np.linalg.norm(g)

    while f(x - (t*g).flatten()) > f(x) - eta*t*norm and k < itmax:
        
        t = 0.5*t

        k = k + 1

    print("LINE SEARCH: t = ", t, " iterations = ", k)
    
    return t, k