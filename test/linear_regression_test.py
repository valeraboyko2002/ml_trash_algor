import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from algorithms.supervised.linear_regression import LinearRegression

X,y = datasets.make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

fig = plt.figure(figsize=(8,6))
plt.scatter(X,y, color='b', marker='o', s=30)
plt.title("Сгенерированные данные для линейной регрессии")
plt.xlabel("Фича")
plt.ylabel("Целевая переменная")
plt.show()

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_value = mse(y_test, y_pred)
print(f"MSE на тестовом наборе: {mse_value:.2f}")

y_pred_line = model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Линейная регрессия')
plt.show()