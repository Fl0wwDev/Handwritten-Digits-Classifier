import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

digits = load_digits()

scaler = StandardScaler()
x_normalized = scaler.fit_transform(digits.data)
model = LogisticRegression(max_iter=1000)
model.fit(x_normalized, digits.target)
predictions = model.predict(x_normalized)
accuracy = np.mean(predictions == digits.target)
print(f"\nPrécision: {accuracy * 100:.2f}%")