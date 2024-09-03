#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import numpy as np

import math

import pseudoinverse, gen_simp_grad, gen_simp_hessian


def newton(delta, x_0, D, f, e_stop, itmax):
    
    k = 0

    x = 0

    stop = False

    # MODEL CREATION PHASE

    # using a finite number of points to create a model for f with order-1 gradient accuracy

    while stop == False:

        print(" --- ITERATION #", k, " ---")

        D = delta * D

        g = gen_simp_grad.gen_simplex_gradient(x_0, D, f)

        # MODEL ACCURACY CHECKS

        norm = np.linalg.norm(g,ord = np.inf)

        print("delta =", delta)

        print("g =", g)

        H = gen_simp_hessian.gen_simplex_hessian(x_0, D, D, f)

        x = x_0 - np.dot(np.linalg.inv(H), g)

        print("x = ", x)

        print("step size ", np.linalg.norm(x_0 - x))


        if np.linalg.norm(x_0 - x, ord = np.inf) < e_stop:

            # algorithm succeds and we stop

            stop = True

            print("success")

            print(x)

            return x

        elif k > itmax:

            print("exceeded iterations")

        x_0 = x

        delta = 0.5 * delta

        k = k + 1

    return x



def r(x):

    return pow(x[0],2) + pow(x[1], 2)


#def newton(delta, x_0, D, f, e_stop, itmax):

newton(1e-1,np.array([1, 2]), np.array([[1,0], [0, 1]]), r, 1e-4, 8)


