from descente_stochastique import GradientDescent
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
# Visualisation d'un chiffre du dataset des chiffres manuscrits
# plt.matshow(digits.images[1], cmap='gray')
# plt.show()

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def gradient_descent_update(params: np.ndarray, data: tuple) -> tuple[np.ndarray, None]:
    x, y = data
    weights = params[:-1]
    bias = params[-1]
    
    m = x.shape[0]
    sigma = sigmoid(np.dot(x, weights) + bias)
    dW = 1/m * np.dot(x.T, (sigma - y))
    db = 1/m * np.sum(sigma - y)
    
    return np.concatenate([dW, [db]]), None

def optimize(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    initial_point = np.concatenate([np.zeros(x.shape[1]), [0.0]])
    
    gd = GradientDescent(gradient=gradient_descent_update, learning_rate=0.1, max_iterations=1000, epsilon=1e-6, batch_size=32)
    optimized_params = gd.descent(initial_point=initial_point, data=(x, y))
    
    return optimized_params[:-1], optimized_params[-1]

# régression logistique multi-classes (One-vs-Rest)
all_weights = []
all_biases = []

for digit in range(10):
    print(f"Entraînement pour le chiffre {digit}...")
    y_binary = (digits.target == digit).astype(int)
    weights, bias = optimize(digits.data, y_binary)
    all_weights.append(weights)
    all_biases.append(bias)

def predict(x: np.ndarray, all_weights: list, all_biases: list) -> np.ndarray:
    probabilities = [sigmoid(np.dot(x, w) + b) for w, b in zip(all_weights, all_biases)]
    probabilities = np.array(probabilities).T
    return np.argmax(probabilities, axis=1)

predictions = predict(digits.data, all_weights, all_biases)
accuracy = np.mean(predictions == digits.target)
print(f"\nPrécision: {accuracy * 100:.2f}%")