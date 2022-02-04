import numpy as np
from linear_regression import LinearRegression

class lr_SGD(LinearRegression):

    def __init__ (self, learning_rate, n_iters):
        super().__init__(learning_rate, n_iters)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        id = np.random.permutation(n_samples)

        for epoch in range(n_samples):
            x_i = X[id[epoch], :]
            y_i = y[id[epoch]]

            y_predicted = np.dot(x_i, self.weights) + self.bias
            error = y_predicted - y_i

            dw= x_i * error
            db= error
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            cost = error**2
            self.costs.append(cost)

            print(f'epoch: {epoch+1}, '
            f'weights = {self.weights[0]}, '
            f'bias = {self.bias}, '
            f'cost = {cost}')
