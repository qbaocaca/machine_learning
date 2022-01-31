import numpy as np
import matplotlib.pyplot as plt

from logistic_regression import create_data, visualize_data
from logistic_regression import LogisticRegression
from logistic_regression import visualize_cost
from logistic_regression import compute_loss
from logistic_regression import accuracy

X1 = create_data(20, -3, 5, -2, 4, 0, 1)
X2 = create_data(20, 5, 8, 4, 9, 1, 2)
X = np.concatenate((X1, X2))
y = X[:, 2]
y = np.array([int(i) for i in y])

visualize_data(X, y)

X = np.delete(X, slice(2, 3), 1)
np.random.shuffle(X)

model = LogisticRegression(learning_rate=0.01, n_iters=10000)
model.fit(X, y)

visualize_cost(model)

print(f'LR classification loss: {compute_loss(X, y, model)}')

X1_test = create_data(100, -7, 2, -3, 5, 0, 1)
X2_test = create_data(100, -14, -8, -8, -4, 1, 2)
X_test = np.concatenate((X1_test, X2_test))
y_test = X_test[:, 2]
y_test = np.array([int(i) for i in y_test])

visualize_data(X_test, y_test)

X_test = np.delete(X_test, slice(2, 3), 1)
np.random.shuffle(X_test)

predictions = model.predict(X_test)
print(f'Test accuracy: {accuracy(y_test, predictions)* 100}%')
