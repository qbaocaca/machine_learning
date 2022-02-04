import numpy as np
from logistic_regression import create_data, visualize_data
from softmax_regression import SoftmaxRegression
from logistic_regression import visualize_cost, accuracy

X1 = create_data(20, -1, 2, 1, 3, 0, 1)
X2 = create_data(20, 4, 6, -4, -1, 1, 2)
X3 = create_data(20, -5, -3, -2.5, -0.5, 2, 3)

X = np.concatenate((X1, X2, X3))
np.random.shuffle(X)

y = X[:, 2]
y = np.array([int(i) for i in y])

X = np.delete(X, slice(2, 3), 1)

# visualize_data(X, y)

model = SoftmaxRegression(learning_rate=0.1, n_iters=1000, num_classes=3)
model.fit(X, y)

# visualize_cost(model)

X1_test = create_data(15, -1, 2, 1, 3, 0, 1)
X2_test = create_data(15, 4, 6, -4, -1, 1, 2)
X3_test = create_data(15, -5, -3, -2.5, -0.5, 2, 3)

X_test = np.concatenate((X1_test, X2_test, X3_test))
np.random.shuffle(X_test)

y_test = X_test[:, 2]
y_test = np.array([int(i) for i in y_test])

X_test = np.delete(X_test, slice(2, 3), 1)

visualize_data(X_test, y_test)

y_pred = model.predict(X_test)
print(f'Test accuracy: {accuracy(y_test, y_pred)* 100}%')
