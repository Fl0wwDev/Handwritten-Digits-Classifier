import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_files(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    return np.genfromtxt(file_path, delimiter=',')

x_data_train = load_files("data_ex2/x_train.csv")
y_data_train = load_files("data_ex2/y_train.csv")
x_data_test = load_files("data_ex2/x_test.csv")
y_data_test = load_files("data_ex2/y_test.csv")

# Q2
plt.figure(figsize=(10, 6))
plt.scatter(x_data_train[y_data_train == 0, 0], x_data_train[y_data_train == 0, 1], color='blue', label='Not Iris-Virginica', alpha=0.5)
plt.scatter(x_data_train[y_data_train == 1, 0], x_data_train[y_data_train == 1, 1], color='red', label='Iris-Virginica', alpha=0.5)
plt.legend()
plt.title('Données d\'entraînement')
plt.grid()
plt.show()

# Q4
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(x_data_train)
X_test_norm = scaler.transform(x_data_test)
model = LogisticRegression()
model.fit(X_train_norm, y_data_train)

print("\nModèle de régression logistique:")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
train_accuracy = model.score(X_train_norm, y_data_train)
test_accuracy = model.score(X_test_norm, y_data_test)
print(f"Exactitude sur les données d'entraînement: {train_accuracy}")
print(f"Exactitude sur les données de test: {test_accuracy}")

# Q5
x_min, x_max = X_train_norm[:, 0].min() - 1, X_train_norm[:, 0].max() + 1
y_min, y_max = X_train_norm[:, 1].min() - 1, X_train_norm[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_train_norm[y_data_train == 0, 0], X_train_norm[y_data_train == 0, 1], color='blue', label='Not Iris-Virginica', alpha=0.5)
plt.scatter(X_train_norm[y_data_train == 1, 0], X_train_norm[y_data_train == 1, 1], color='red', label='Iris-Virginica', alpha=0.5)
plt.legend()
plt.title('Frontière de décision et données d\'entraînement')
plt.grid()
plt.show()