import numpy as np 

class TinyNeuralNetwork:
    def __init__(self):
        np.random.seed(42)
        self.weights = 2 * np.random.random((3,1))-1

    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))
    
    def train(self,inputs,outputs,iterations=10000):
        for i in range(iterations):
            # прямое распространение
            output = self.predict(inputs)

            # вычисление ошибки
            error = outputs - output

            #корректировка весов
            adjustments = np.dot(inputs.T, error * output * (1 - output))
            self.weights += adjustments
    
    def predict(self,inputs):
        return self.sigmoid(np.dot(inputs, self.weights))

# Пример использования
if __name__ == "__main__":
    # Создание набора данных
    np.random.seed(0)
    inputs = np.random.rand(1000, 3)
    outputs = (inputs.sum(axis=1) > 1.5).astype(int).reshape(-1, 1)

    # Инициализация и обучение нейронной сети
    nn = TinyNeuralNetwork()
    nn.train(inputs, outputs, iterations=10000)

    # Тестирование нейронной сети
    test_input = np.array([[0.1, 0.2, 0.3],
                           [0.5, 0.5, 0.5],
                           [0.9, 0.9, 0.9]])
    predictions = nn.predict(test_input)
    print("Predictions:")
    print(predictions)

# ______________________________________________________________________________________________________________________________________________________
# визуализация процесса обучения 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
plt.ion()
losses = []
for i in range(1000):
    nn.train(X_train, y_train, iterations=1)
    train_output = nn.predict(X_train)
    loss = np.mean((y_train - train_output) ** 2)
    losses.append(loss)

    if i % 100 == 0:
        plt.clf()
        plt.plot(losses)
        plt.xlabel('Кол-во итераций')
        plt.ylabel('потери')
        plt.title('потери на протяжении времени')
        plt.pause(0.1)
plt.ioff()
plt.show()

# ___________________________________________________________________________________________________________________________________________________________
# визуализация сигмоидной функции
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# значения от -10 до 10
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(10, 4))
plt.plot(x, y, 'b-', linewidth=3)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Вход (x)')
plt.ylabel('Выход сигмоиды')
plt.title('Сигмоидная функция')
plt.show()