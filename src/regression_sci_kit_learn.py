import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

class LogisticRegressionSklearn:
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.digits = load_digits()
        self.x_normalized = self.scaler.fit_transform(self.digits.data)
        self.model = LogisticRegression(max_iter=self.max_iter)
        self.model.fit(self.x_normalized, self.digits.target)

    def predict(self, x):
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled)

    def get_accuracy(self):
        predictions = self.predict(self.digits.data)
        return np.mean(predictions == self.digits.target)
    
    def show_predictions_plt(self, num_samples=10):
        
        sample_indices = np.random.choice(len(self.digits.data), num_samples, replace=False)
        samples = self.digits.data[sample_indices]
        true_labels = self.digits.target[sample_indices]
        predicted_labels = self.predict(samples)
        
        plt.figure(figsize=(10, 4))
        for i, index in enumerate(sample_indices):
            plt.subplot(2, num_samples // 2, i + 1)
            plt.imshow(self.digits.images[index], cmap='gray')
            plt.title(f'True: {true_labels[i]}\nPred: {predicted_labels[i]}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    model = LogisticRegressionSklearn()
    accuracy = model.get_accuracy()
    print(f"Précision du modèle scikit-learn: {accuracy * 100:.2f}%")
    model.show_predictions_plt()