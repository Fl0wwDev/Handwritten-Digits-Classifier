# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:47:15 2024

@author: JDION
"""

import numpy as np

from my_descent_td1 import GradientDescent


### EXERCICE 5
print("Exercice 5")

def f_ex5(x):
    return x**2 + 1

def gradient_f_ex5(x):
    return 2*x

ex5_a = GradientDescent(gradient_f_ex5, 0.2, 10)
image = ex5_a.descent(2)
print("a) :", image)

ex5_b = GradientDescent(gradient_f_ex5, 0.01, 10)
print("b) :", ex5_b.descent(-1.5))

ex5_c1 = GradientDescent(gradient_f_ex5, 0.9, 10)
print("c) 1 :", ex5_c1.descent(2))

ex5_c2 = GradientDescent(gradient_f_ex5, 1.1, 10)
print("c) 2 :", ex5_c2.descent(2))

ex5_c3 = GradientDescent(gradient_f_ex5, 0.05, 10)
print("c) 3 :", ex5_c3.descent(2))

### EXERCICE 6
print("\nExercice 6")

def f_ex6(x):
    return x**4 - 2*x**3 + 4

def gradient_f_ex6(x):
    return 4*x**3 - 6*x**2

ex6_a = GradientDescent(gradient_f_ex6, 0.05, 1000)
print("a) :", ex6_a.descent(-1))

### EXERCICE 7 
print("\nExercice 7")

def f_ex7(x, y):
    return x ** 2 + 3 * y ** 2

def gradient_f_ex7(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    dfdx = 2 * x
    dfdy = 6 * y
    return (dfdx, dfdy)

ex7_a = GradientDescent(gradient_f_ex7, 0.2, 100, 0.01)
print("a) :", ex7_a.descent((1, 1)))
print("Nombre d'itérations :", ex7_a.num_iterations)


ex7_b = GradientDescent(gradient_f_ex7, 0.2, 100, 0.01)
print("b) :", ex7_b.descent((-1, -1)))
print("Nombre d'itérations :", ex7_b.num_iterations)

ex7_c = GradientDescent(gradient_f_ex7, 0.2, 100, 0.01)
print("a) :", ex7_c.descent((2, 1)))
print("Nombre d'itérations :", ex7_c.num_iterations)

### EXERCICE 8

data = np.array(((4, 1), (7, 3), (8, 3), (10, 6), (12, 7)))


### EXERCICE 9

data = np.array(((1, 0, 0), (0, 1, 5), (2, 1, 1), (1, 2, 0), (2, 2, 3)))




