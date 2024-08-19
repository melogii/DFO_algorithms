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

    print(hessian)

    return hessian


def p(x):

    return x[0]*x[0]*x[0] + x[1]*x[1]*x[1] + x[2]*x[2]*x[2]

x = np.array([2, -1, 1])

D = np.array([[0.1, 0, 0, -0.1, 0], [0, 0.1, -0.1, -0.2, 0.2], [-0.1, 0, 0.1, 0.2, -0.2]])

D = 0.01 * D

print(gen_simplex_hessian(x, D, D, p))

def mbsd(delta, mi, eta, e_stop, x, D, f, itmax):
    
    k = 0

    # MODEL CREATION PHASE

    # using a finite number of points to create a model for f with order-1 gradient accuracy

    D = delta * D

    g = gen_simplex_gradient(x, D, f)

    # MODEL ACCURACY CHECKS

    norm = np.linalg.norm(g,ord = np.inf)

    if delta < e_stop and norm:

        # algorithm succeds and we stop

        return x

    elif delta > mi * norm:

        # model is innacurate so we do the following

        delta = 0.5 * delta

        # keep mi and x as they are

        k = k + 1

    elif delta <= mi * norm:

        # model is acurate and therefore we proceed to the nex step

        #LINE SEARCH

        t = line_search(f,x, g, eta, itmax)[0]

        # We are declaring succes and failure based on the iteration count

        if line_search(f,x, g, eta, itmax)[1] < itmax:

            # update x at iteration k

            x = x - t*g

        else:

            mi = 0.5 * mi

        k = k + 1


    return


def line_search(f,x, g, eta, itmax):

    t = 1

    k = 1

    norm = np.linalg.norm(g)

    while f(x - t*g) > f(x) - eta*t*norm and k < itmax:
        
        t = 0.5*t

        k = k + 1
    
    return t, k
    

def l(x):

    return x[0]**2 + 2*x[1]**2

def grad_l(x):

    return np.array([2*x[0], 4*x[1]])

x = np.array([1.0, 2.0])

g = grad_l(x)

print(line_search(l, x, g, 1e-6, 100)[0])



