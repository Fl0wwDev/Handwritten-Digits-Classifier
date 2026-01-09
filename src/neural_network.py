import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

class NeuralNetworkCustom:
    def __init__(self, dataset, digits_data, digits_target, image_shape, epochs=3):
        self.dataset = dataset
        self.digits_data = digits_data
        self.digits_target = digits_target
        self.image_shape = image_shape
        self.epochs = epochs
        # Build a minimal neural network
        self.model = Sequential([
            Flatten(input_shape=image_shape),
            Dense(16, activation='relu'),
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self):
        X_train = self.digits_data.reshape(-1, *self.image_shape) / 255.0
        self.model.fit(X_train, self.digits_target, epochs=self.epochs, verbose=1)

    def predict(self, x):
        x_reshaped = x.reshape(-1, *self.image_shape) / 255.0
        preds = self.model.predict(x_reshaped, verbose=0)
        return np.argmax(preds, axis=1)

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
            image = samples[i].reshape(self.image_shape)
            plt.imshow(image, cmap='gray')
            plt.title(f'Réalité: {true_labels[i]}\nPrédiction: {predicted_labels[i]}')
            plt.axis('off')
        plt.suptitle('Prédictions du Neural Network')
        plt.tight_layout()
        plt.show()

    def confusion_matrix_display(self):
        predictions = self.predict(self.digits_data)
        cm = confusion_matrix(self.digits_target, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Matrice de confusion - Neural Network')
        plt.show()

if __name__ == "__main__":
    (X_train, y_train), _ = mnist.load_data()
    dataset = (X_train, y_train)
    digits_data = X_train.reshape(-1, 28*28)
    digits_target = y_train
    image_shape = (28, 28)
    model = NeuralNetworkCustom(dataset=dataset, digits_data=digits_data, digits_target=digits_target, image_shape=image_shape)
    model.fit()
    accuracy = model.get_accuracy()
    print(f"Précision du modèle Neural Network: {accuracy * 100:.2f}%")
    model.show_predictions_plt()