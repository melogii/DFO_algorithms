#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import numpy as np

import math

import gen_simp_grad, line_search

def mbsd(delta, mi, eta, e_stop, x, D, f, itmax):
    
    k = 0

    stop = False

    # MODEL CREATION PHASE

    # using a finite number of points to create a model for f with order-1 gradient accuracy

    while stop == False:

        print(" --- ITERATION #", k, " ---")

        D = delta * D

        g = gen_simp_grad.gen_simplex_gradient(x, D, f)

        # MODEL ACCURACY CHECKS

        norm = np.linalg.norm(g,ord = np.inf)

        print("delta =", delta)

        print("g =", g)

        print("norm = ", norm)

        if delta < e_stop and norm < e_stop:

            # algorithm succeds and we stop

            stop = True

            print("success")

            print(x)

            return x

        elif delta > mi * norm:

            # model is innacurate so we do the following

            delta = 0.5 * delta

            # keep mi and x as they are

            print("model innacurate")

            k = k + 1

        elif delta <= mi * norm:

            print("model accurate and proceed")

            # model is acurate and therefore we proceed to the nex step

            #LINE SEARCH

            print("old_x = ", x)

            t, iterations = line_search.line_search(f, x, g, eta, itmax)

            # We are declaring succes and failure based on the iteration count

            if iterations < itmax:

                # update x at iteration k

                x = x - t*g

                print("new_x = ", x)

            else:

                mi = 0.5 * mi

            k = k + 1

    return x


#mbsd(delta, mi, eta, e_stop, x, D, f, itmax)

def r(x):

    return pow(x[0],2) + pow(x[1], 2)

#mbsd(1e-1, 1e-3, 1/2, 1e-4, np.array([1, 2]), np.array([[1,0], [0, 1]]), r, 8)

#mbsd(1e-1, 1, 1/2, 1e-6, np.array([1, 1]), np.array([[1,0], [0, 1]]), r, 8) # SEQUENCE OF MODEL INNACCURATE

#mbsd(1e-2, 1e-3, 1/2, 1e-6, np.array([1, 1]), np.array([[1,0], [0, 1]]), r, 8) # one iteration of line search

def s(x):

    return np.sqrt(pow(x[0],2) + pow(x[1], 2))

#mbsd(1e-2, 1e-3, 1/2, 1e-6, np.array([1, 1]), np.array([[1,0], [0, 1]]), s, 8) # one iteration of line search