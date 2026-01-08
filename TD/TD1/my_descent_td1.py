# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:45:29 2024

@author: JDION
"""

"""
Création de la classe GradientDescent utilisée durant l'intégralité de ce cours
"""
import numpy as np
class GradientDescent:
    
    def __init__(self, gradient, learning_rate = 0.01, max_iterations = 1000, error_rate = 0.01):
        
        """
        Initialise l'objet GradientDescent avec les paramètres nécessaires.

        Paramètres :
        - gradient : La fonction gradient de la fonction de coût.
        - learning_rate : Taux d'apprentissage (pas) pour la mise à jour des paramètres.
        - max_iterations : Nombre maximal d'itérations de l'algorithme de descente.
        """
        
        self.gradient = gradient
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.error_rate = error_rate
        self.num_iterations = 0

    def descent(self, initial_point: float | tuple[float, float]) -> float | tuple[float, float]:
        
        """
        Effectue l'algorithme de descente de gradient.

        Paramètres :
        - initial_point : Le point de départ de l'algorithme.

        Retourne :
        - Le point optimal trouvé par l'algorithme.
        """
        for _ in range(self.max_iterations):
            gradient_value = self.gradient(initial_point)
            if np.linalg.norm(gradient_value) < self.error_rate:
                break
            initial_point = self.update(initial_point, gradient_value)
            self.num_iterations += 1
        return initial_point

    def update(self, point, gradient_value):
        
        """
        Met à jour le point en utilisant le gradient et le taux d'apprentissage.

        Paramètres :
        - point : Le point à mettre à jour.
        - gradient_value : Le gradient de la fonction de coût au point donné.

        Retourne :
        - Le nouveau point après la mise à jour.
        """
        # ak+1 = ak −δk∇(E)(ak)
        if isinstance(point, tuple):
            point_x, point_y = point
            gradient_x, gradient_y = gradient_value
            new_point_x = point_x - self.learning_rate * gradient_x
            new_point_y = point_y - self.learning_rate * gradient_y
            new_point = (new_point_x, new_point_y)
        else:
            new_point = point - self.learning_rate * gradient_value 
        return new_point


