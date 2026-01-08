import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

with open(os.path.join(os.path.dirname(__file__), "data.csv"), 'r') as file:
    data = file.readlines()
    data = [list(map(float, line.strip().split(','))) for line in data]
    x = np.array(data[0])
    y = np.array(data[1])

############### EXERCICE 2 ##############

plt.scatter(x, y, color='blue', label='Données')

poly = PolynomialFeatures(degree=4, include_bias=False)
x_reshaped = x.reshape(-1, 1)
x_poly = poly.fit_transform(x_reshaped)

model = LinearRegression()
model.fit(x_poly, y)

x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_pred = model.predict(x_range_poly)

plt.plot(x_range, y_pred, color='red', label='Ajustement polynomial degré 4')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print("Coefficients du modèle polynomial :", model.coef_)
print("Intercept :", model.intercept_)
print("Erreur quadratique moyenne (MSE) :", mean_squared_error(y, model.predict(x_poly)))

############### EXERCICE 3 ##############

housing = fetch_california_housing()
X, y = housing.data, housing.target

# Q2a
reg = LinearRegression().fit(X, y)
print("intercept :", reg.intercept_)
print("coefficients :", reg.coef_)

# Q2b
prediction = reg.predict(X)
print("prediction groupe 1:", prediction[0])
print("valeur réelle groupe 1:", y[0])

# Q2c
diff = np.abs(y - prediction)
group_num = np.argmin(diff)
print('Numéro du groupe:', group_num)
print('Prédiction du modèle pour ce groupe:', prediction[group_num - 1])
print('Valeur réelle du groupe:', y[group_num - 1])
