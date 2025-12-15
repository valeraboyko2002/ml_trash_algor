import sys
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from algorithms.supervised.knn import KNN

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

model = KNN(k=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
accuracy_manual = np.mean(y_pred == y_test)

print("Результаты KNN:")
print(f"Предсказания (первые 10): {y_pred[:10]}")
print(f"Фактические (первые 10): {y_test[:10]}")
print(f"Точность модели: {accuracy:.2%}")
print(f"Правильных предсказаний: {np.sum(y_pred == y_test)} из {len(y_test)}")