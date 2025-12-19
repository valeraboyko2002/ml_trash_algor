import numpy as np
from sklearn.datasets import load_breast_cancer
from algorithms.supervised.logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split


data = load_breast_cancer()
X,y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1, test_size=.33)


model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


def score(y_true, y_pred):
    return np.sum(y_pred == y_true) / len(y_true)

acc = score(y_test, y_pred)
print(f"точность =  {acc*100:.2f}%")