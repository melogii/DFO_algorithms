#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import unittest

import DFO

import numpy as np


class TestCalc(unittest.TestCase):

    def test_f(self):

        self.assertAlmostEqual(DFO.f(2), 4, places = 8)
        self.assertAlmostEqual(DFO.f(-1), 1, places = 8)
        self.assertAlmostEqual(DFO.f(0), 0, places = 8)

    def test_g(self):

        self.assertAlmostEqual(DFO.g(1,1), np.sqrt(2), places = 8)
        self.assertAlmostEqual(DFO.g(-1,1), np.sqrt(2), places = 8)
        self.assertAlmostEqual(DFO.g(-1,-1), np.sqrt(2), places = 8)

    def test_h(self):

        self.assertAlmostEqual(DFO.h(1,1,1), 1/2, places = 8)
        self.assertAlmostEqual(DFO.h(1,0,1), 1, places = 8)
        self.assertAlmostEqual(DFO.h(-1,-1,0), 1, places = 8)

    def test_simplex_gradient(self):

        def m(x):

            return x[0]*x[0] + 2*x[1] - 3
        
        for t in range(2):

            self.assertAlmostEqual(DFO.simplex_gradient([0, 0], np.array([[2, 2], [1, -2]]), m)[t][0], np.array([[2],[2]])[t][0], places = 8)

        for t in range(2):

            self.assertAlmostEqual(DFO.simplex_gradient([0, 0], np.array([[0.02, 0.02], [0.01, -0.02]]), m)[t][0], np.array([[0.02], [2]])[t][0], places = 8)


    def test_pseudoinverse(self): #- look for a more efficient way of comparing matrices

        A = np.array([[1/3, 2, 0], [2, -6, 0]])

        np.testing.assert_allclose(DFO.pseudoinverse(A), np.array([[1, 1/3], [1/3, -1/18], [0, 0]]), atol=1e-08)

        B = np.array([[1/3, 2, 0], [2, -6, 0], [0, 0, -4]])

        np.testing.assert_allclose(DFO.pseudoinverse(B), np.array([[1, 1/3, 0], [1/3, -1/18, 0], [0, 0, -1/4]]), atol=1e-08)


    def test_gen_simplex_gradient(self):

        def m(x):

            return x[0]*x[0] + 2*x[1] - 3
        
        D = np.array([0.02, 0.01])
        
        D = D.reshape(-1, 1)

        for t in range(2):

            self.assertAlmostEqual(DFO.gen_simplex_gradient([0, 0], D, m)[t][0], np.array([[0.8160],[0.4080]])[t][0], places = 8)

        D = np.array([[0.02, 0.02, -0.02], [0.01, -0.02, -0.01]])

        for t in range(2):

            self.assertAlmostEqual(DFO.gen_simplex_gradient([0, 0], D, m)[t][0], np.array([[0.006666666],[1.986666666]])[t][0], places = 8)


    def test_gen_simplex_hessian(self):

        def p(x):

            return x[0]*x[0]*x[0] + x[1]*x[1]*x[1] + x[2]*x[2]*x[2]
        
        x = np.array([2, -1, 1])
        
        D = np.array([[0.1, 0, 0, -0.1, 0], [0, 0.1, -0.1, -0.2, 0.2], [-0.1, 0, 0.1, 0.2, -0.2]])

        D = 0.001 * D

        np.testing.assert_allclose(DFO.gen_simplex_hessian(x, D, D, p),np.array([[12, 0, 0], [0, -6, 0], [0, 0, 6]]), atol=1e-03)


        def q(x):

            return x[0]*np.exp(x[0] + x[1])
        
        x = np.array([0,0])
        
        D = np.array([[0.1, 0], [0, 0.1]])

        np.testing.assert_allclose(DFO.gen_simplex_hessian(x, D, D, q),np.array([[2, 1], [1, 0]]), atol=1e-08)





if __name__ == '__main__':
    unittest.main()
