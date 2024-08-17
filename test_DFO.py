#! C:\Users\giova\OneDrive\numpy_project\myenv\Scripts\python.exe

import unittest

import DFO

import numpy as np


class TestCalc(unittest.TestCase):


    def test_f(self):

        self.assertEqual(DFO.f(2), 4)
        self.assertEqual(DFO.f(-1), 1)
        self.assertEqual(DFO.f(0), 0)

    def test_g(self):

        self.assertEqual(DFO.g(1,1), np.sqrt(2))
        self.assertEqual(DFO.g(-1,1), np.sqrt(2))
        self.assertEqual(DFO.g(-1,-1), np.sqrt(2))

    def test_h(self):

        self.assertEqual(DFO.h(1,1,1), 1/2)
        self.assertEqual(DFO.h(1,0,1), 1)
        self.assertEqual(DFO.h(-1,-1,0), 1)

    def test_simplex_gradient(self):

        def m(x):

            return x[0]*x[0] + 2*x[1] - 3
        
        for t in range(2):

            self.assertAlmostEqual(DFO.simplex_gradient([0, 0], np.array([[2, 2], [1, -2]]), m)[t][0], np.array([[2],[2]])[t][0])

        for t in range(2):

            self.assertAlmostEqual(DFO.simplex_gradient([0, 0], np.array([[0.02, 0.02], [0.01, -0.02]]), m)[t][0], np.array([[0.02], [2]])[t][0])


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

            self.assertAlmostEqual(DFO.gen_simplex_gradient([0, 0], D, m)[t][0], np.array([[0.8160],[0.4080]])[t][0])

        D = np.array([[0.02, 0.02, -0.02], [0.01, -0.02, -0.01]])

        for t in range(2):

            self.assertAlmostEqual(DFO.gen_simplex_gradient([0, 0], D, m)[t][0], np.array([[0.0067],[1.9867]])[t][0])




if __name__ == '__main__':
    unittest.main()
