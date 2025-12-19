from algorithms.supervised.decision_trees import DecisionTreeClassifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X,y = data.data, data.target 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=1)

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
accuracy