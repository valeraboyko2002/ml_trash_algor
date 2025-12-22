# Machine Learning Algorithms from Scratch

Репозиторий с реализацией основных алгоритмов машинного обучения **с нуля** на чистом Python.

## Цель проекта

Разобраться в фундаментальных принципах работы алгоритмов машинного обучения через их самостоятельную реализацию без использования специализированных библиотек (кроме NumPy для базовых операций).

## 📁 Структура проекта

```
ml-from-scratch/
├── algorithms/           # Реализации алгоритмов
│   ├── supervised/      # Алгоритмы с учителем
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── knn.py
│   ├── unsupervised/    # Алгоритмы без учителя
├── test/              # Unit-тесты
└── README.md
```

## Алгоритмы

### Алгоритмы с учителем
- **Линейная регрессия** - с градиентным спуском и нормальным уравнением
- **Логистическая регрессия** - для бинарной и многоклассовой классификации
- **Метод k-ближайших соседей (k-NN)**
- **Наивный байесовский классификатор**
- **Дерево решений** - с критериями Джини и энтропии
- **Случайный лес** - ансамбль деревьев решений
- **Метод опорных векторов (SVM)** - с ядрами
- **Градиентный бустинг**
- **Нейронные сети** - MLP, CNN (базовая реализация)

### Алгоритмы без учителя
- **K-средних (K-Means)** - кластеризация
- **Иерархическая кластеризация**
- **Метод главных компонент (PCA)** - снижение размерности
- **DBSCAN** - плотностная кластеризация
- **Алгоритм Apriori** - поиск ассоциативных правил

### Вспомогательные модули
- **Метрики** - точность, полнота, F1, MSE, R² и др.
- **Предобработка** - нормализация, кодирование, разделение выборки
- **Визуализация** - графики обучения, границы решений


### Базовые зависимости
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scikit-learn (только для сравнения и загрузки датасетов)
- jupyter (для работы с ноутбуками)

### Пример использования
```python
from algorithms.supervised.linear_regression import LinearRegression
from utils.metrics import mean_squared_error
from utils.preprocessing import train_test_split

# Создание и обучение модели
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Предсказание
predictions = model.predict(X_test)

# Оценка
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse:.4f}")
```

## Ресурсы

- [Scikit-learn документация](https://scikit-learn.org/)
- [Библиотека NumPy](https://numpy.org/)
- [Курс Andrew Ng на Coursera](https://www.coursera.org/learn/machine-learning)
- ["Глубокое обучение" Гудфеллоу, Бенджио, Курвилль](http://www.deeplearningbook.org/)

## 📄 Лицензия

MIT License - подробности в файле [LICENSE](LICENSE)

---

