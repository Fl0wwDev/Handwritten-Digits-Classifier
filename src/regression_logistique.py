from descente_stochastique import GradientDescent
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.1, max_iterations=1000, epsilon=1e-6, batch_size=32, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.all_weights = []
        self.all_biases = []
        self.scaler = StandardScaler()
        self.digits = load_digits()
        self.x_normalized = self.scaler.fit_transform(self.digits.data)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient_descent_update(self, params, data):
        x, y = data
        weights = params[:-1]
        bias = params[-1]
        
        m = x.shape[0]
        sigma = self.sigmoid(np.dot(x, weights) + bias)
        dW = 1/m * np.dot(x.T, (sigma - y))
        db = 1/m * np.sum(sigma - y)
        
        dW += self.lambda_reg * weights
        
        return np.concatenate([dW, [db]]), None

    def optimize(self, x, y):
        initial_point = np.concatenate([np.zeros(x.shape[1]), [0.0]])
        
        gd = GradientDescent(gradient=self.gradient_descent_update, learning_rate=self.learning_rate, 
                             max_iterations=self.max_iterations, epsilon=self.epsilon, batch_size=self.batch_size)
        optimized_params = gd.descent(initial_point=initial_point, data=(x, y))
        
        return optimized_params[:-1], optimized_params[-1]

    def fit(self):
        for digit in range(10):
            print(f"Entraînement pour le chiffre {digit}...")
            y_binary = (self.digits.target == digit).astype(int)
            weights, bias = self.optimize(self.x_normalized, y_binary)
            self.all_weights.append(weights)
            self.all_biases.append(bias)

    def predict(self, x):
        probabilities = [self.sigmoid(np.dot(x, w) + b) for w, b in zip(self.all_weights, self.all_biases)]
        probabilities = np.array(probabilities).T
        return np.argmax(probabilities, axis=1)

    def get_accuracy(self):
        predictions = self.predict(self.x_normalized)
        return np.mean(predictions == self.digits.target)
    
    def show_predictions_plt(self, num_samples=10):
        sample_indices = np.random.choice(len(self.digits.data), num_samples, replace=False)
        samples = self.digits.data[sample_indices]
        true_labels = self.digits.target[sample_indices]
        predicted_labels = self.predict(self.scaler.transform(samples))
        
        plt.figure(figsize=(10, 4))
        for i, index in enumerate(sample_indices):
            plt.subplot(2, num_samples // 2, i + 1)
            plt.imshow(self.digits.images[index], cmap='gray')
            plt.title(f'Réalité: {true_labels[i]}\nPrédiction: {predicted_labels[i]}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def confusion_matrix_display(self):
        predictions = self.predict(self.digits.data)
        cm = confusion_matrix(self.digits.target, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Matrice de confusion - Scikit-learn Logistic Regression')
        plt.show()

if __name__ == "__main__":
    model = LogisticRegressionCustom()
    model.fit()
    accuracy = model.get_accuracy()
    print(f"Précision du modèle from-scratch: {accuracy * 100:.2f}%")
    model.show_predictions_plt()