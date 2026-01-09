from descente_stochastique import GradientDescent
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.1, max_iterations=1000, epsilon=1e-6, batch_size=32, lambda_reg=0.01, dataset=load_digits(), digits_data=load_digits().data, digits_target=load_digits().target, image_shape=(8, 8)):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.all_weights = []
        self.all_biases = []
        self.scaler = StandardScaler()
        self.digits = dataset
        self.digits_data = digits_data
        self.digits_target = digits_target
        self.image_shape = image_shape
        self.x_normalized = self.scaler.fit_transform(self.digits_data)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def gradient_descent_update(self, params, data):
        x, y = data
        n_classes = 10
        n_features = x.shape[1]
        weights = params[:n_features * n_classes].reshape(n_features, n_classes)
        biases = params[n_features * n_classes:]
        
        m = x.shape[0]
        logits = np.dot(x, weights) + biases
        probs = self.softmax(logits)
        
        y_onehot = np.zeros((m, n_classes))
        y_onehot[np.arange(m), y] = 1
        
        dW = (1/m) * np.dot(x.T, (probs - y_onehot))
        db = (1/m) * np.sum(probs - y_onehot, axis=0)
        
        dW += self.lambda_reg * weights #test ajout régularisation
        
        return np.concatenate([dW.flatten(), db]), None

    def optimize(self, x, y):
        n_features = x.shape[1]
        n_classes = 10
        initial_point = np.concatenate([np.zeros(n_features * n_classes), np.zeros(n_classes)])
        
        gd = GradientDescent(gradient=self.gradient_descent_update, learning_rate=self.learning_rate, 
                             max_iterations=self.max_iterations, epsilon=self.epsilon, batch_size=self.batch_size)
        optimized_params = gd.descent(initial_point=initial_point, data=(x, y))
        
        weights = optimized_params[:n_features * n_classes].reshape(n_features, n_classes)
        biases = optimized_params[n_features * n_classes:]
        return weights, biases

    def fit(self):
        print("Entraînement du modèle multiclasse avec softmax...")
        self.all_weights, self.all_biases = self.optimize(self.x_normalized, self.digits_target)

    def predict(self, x):
        logits = np.dot(x, self.all_weights) + self.all_biases
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)

    def get_accuracy(self):
        predictions = self.predict(self.x_normalized)
        return np.mean(predictions == self.digits_target)
    
    def show_predictions_plt(self, num_samples=10):
        sample_indices = np.random.choice(len(self.digits_data), num_samples, replace=False)
        samples = self.digits_data[sample_indices]
        true_labels = self.digits_target[sample_indices]
        predicted_labels = self.predict(self.scaler.transform(samples))
        
        plt.figure(figsize=(15, 6))
        for i, index in enumerate(sample_indices):
            plt.subplot(2, num_samples // 2, i + 1)
            if hasattr(self.digits, 'images'):  # For load_digits
                image = self.digits.images[index]
            else:
                image = self.digits_data[index].reshape(self.image_shape)
            plt.imshow(image, cmap='gray')
            plt.title(f'Réalité: {true_labels[i]}\nPrédiction: {predicted_labels[i]}')
            plt.axis('off')
        plt.suptitle('Prédictions du modèle from-scratch')
        plt.tight_layout()
        plt.show()
    
    def confusion_matrix_display(self):
        predictions = self.predict(self.digits_data)
        cm = confusion_matrix(self.digits_target, predictions)
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