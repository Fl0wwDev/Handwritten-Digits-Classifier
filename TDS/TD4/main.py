import matplotlib.pyplot as plt
import os
import numpy as np
from descente_stochastique import GradientDescent
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import fetch_california_housing


with open(os.path.join(os.path.dirname(__file__), "data.csv"), 'r') as file:
    data = file.readlines()
    data = [list(map(float, line.strip().split(','))) for line in data]
    x = np.array(data[0])
    y = np.array(data[1])

degree = 20
x_reshaped = x.reshape(-1, 1)
X = np.column_stack([x_reshaped**i for i in range(1, degree + 1)])  # Degré 1 à 20, sans biais (intercept séparé)

def gradient(theta, batch=None):
    if batch is not None:
        X_batch = X[batch]
        y_batch = y[batch]
    else:
        X_batch = X
        y_batch = y
    predictions = X_batch @ theta
    errors = predictions - y_batch
    return X_batch.T @ errors / len(y_batch)  # Gradient moyen

def plot_regression(x, y, model, poly, scaler):
    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    x_range_poly_scaled = scaler.transform(x_range_poly)
    y_pred = model.predict(x_range_poly_scaled)
    return x_range, y_pred

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=degree, include_bias=False).fit(x_reshaped)

initial_theta = np.zeros(degree)
gd = GradientDescent(gradient=gradient, learning_rate=0.01, max_iterations=10000, epsilon=1e-6, batch_size=10)  # SGD avec batch_size=10

optimal_theta = gd.descent(initial_theta, data=np.arange(len(y)))  # Indices pour mini-batches

# Gradient Descent Predictions
x_range_simple = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
X_range = np.column_stack([x_range_simple**i for i in range(1, degree + 1)])
y_pred_gd = X_range @ optimal_theta

############## EXERCICE 3 ##############
# Ridge Predictions
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_scaled, y)
x_range, y_pred = plot_regression(x, y, ridge_model, poly, scaler)

# Lasso Predictions
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_scaled, y)
x_range_lasso, y_pred_lasso = plot_regression(x, y, lasso_model, poly, scaler)

plt.scatter(x, y, color='blue', label='Données')
plt.plot(x_range_simple, y_pred_gd, color='green', label='Gradient Descent')
plt.plot(x_range, y_pred, color='red', label=f'Ridge (alpha=1.0)')
plt.plot(x_range_lasso, y_pred_lasso, color='orange', label=f'Lasso (alpha=0.1)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

############### EXERCICE 4 ##############
# Régularisation 
housing = fetch_california_housing()
X, y = housing.data, housing.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_lambdas = 200
lambdas = np.logspace(-5, 5, n_lambdas)
coefs = []

for lam in lambdas:
    reg = Lasso(alpha=lam).fit(X_scaled, y)
    result = np.concat(([reg.intercept_], reg.coef_))
    coefs.append(result)
coefs = np.array(coefs)
plt.figure(figsize=(10, 6))
for i in range(coefs.shape[1]):
    plt.plot(lambdas, coefs[:, i], label=f'Coefficient {i}')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Coefficient values')
plt.title('Lasso Paths for California Housing Dataset')
plt.legend()
plt.show()