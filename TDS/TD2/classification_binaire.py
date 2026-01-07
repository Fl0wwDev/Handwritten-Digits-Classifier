import csv
import os
import numpy as np
from descente_stochastique import GradientDescent

def load_data(file_path):
    """
    Charge les données à partir d'un fichier CSV.

    Paramètres :
    - file_path : Le chemin vers le fichier CSV.

    Retourne :
    - Un tableau numpy contenant les données chargées.
    """
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if reader.line_num == 1:
                continue
            data.append([int(value) for value in row])
    return data

data = load_data(os.path.join(os.path.dirname(__file__), "titanic2.csv"))
data = np.array(data)

def f(a, b, x):
    return 1 / (1 + np.exp(-(a * x + b)))

def gradient_cost(theta, mini_batch):
    a, b = theta
    gradient_a = 0.0
    gradient_b = 0.0
    for x_i, y_i in mini_batch:
        prediction = f(a, b, x_i)
        gradient_a += (prediction - y_i) * x_i
        gradient_b += (prediction - y_i)
    return np.array([gradient_a, gradient_b])

gd = GradientDescent(gradient_cost, 0.0001, 100000, 0.001, batch_size=len(data))
result = gd.descent(initial_point=(0.0, 0.0), data=data)
print("Paramètres optimaux (a, b) :", result)