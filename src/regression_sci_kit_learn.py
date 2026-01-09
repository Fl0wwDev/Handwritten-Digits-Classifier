import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

class LogisticRegressionSklearn:
    def __init__(self, max_iterations=100, digits_data=load_digits().data, dataset=load_digits(), digits_target=load_digits().target, image_shape=(8, 8)):
        self.max_iterations = max_iterations
        self.scaler = StandardScaler()
        self.digits = dataset
        self.digits_data = digits_data
        self.digits_target = digits_target
        self.x_normalized = self.scaler.fit_transform(self.digits_data)
        self.model = LogisticRegression(solver="saga", max_iter=self.max_iterations)
        self.model.fit(self.x_normalized, self.digits_target)
        self.image_shape = image_shape

    def predict(self, x):
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled)

    def get_accuracy(self):
        predictions = self.predict(self.digits_data)
        return np.mean(predictions == self.digits_target)
    
    def show_predictions_plt(self, num_samples=10):
        sample_indices = np.random.choice(len(self.digits_data), num_samples, replace=False)
        samples = self.digits_data[sample_indices]
        true_labels = self.digits_target[sample_indices]
        predicted_labels = self.predict(samples)
        
        plt.figure(figsize=(15, 6))
        for i, index in enumerate(sample_indices):
            plt.subplot(2, num_samples // 2, i + 1)
            if hasattr(self.digits, 'images'):
                image = self.digits.images[index]
            else:
                image = self.digits_data[index].reshape(self.image_shape)
            plt.imshow(image, cmap='gray')
            plt.title(f'Réalité: {true_labels[i]}\nPrédiction: {predicted_labels[i]}')
            plt.axis('off')
        plt.suptitle('Prédictions du modèle scikit-learn')
        plt.tight_layout()
        plt.show()


    def confusion_matrix_display(self):
        predictions = self.predict(self.digits_data)
        cm = confusion_matrix(self.digits_target, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Matrice de confusion - Scikit-learn Logistic Regression')
        plt.show()

if __name__ == "__main__":
    model = LogisticRegressionSklearn()
    accuracy = model.get_accuracy()
    print(f"Précision du modèle scikit-learn: {accuracy * 100:.2f}%")
    model.show_predictions_plt()