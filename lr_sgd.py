import numpy as np
from linear_regression import LinearRegression

class lr_SGD(LinearRegression):

    def __init__ (self, learning_rate, n_iters):
        super().__init__(learning_rate, n_iters)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_iters):
            id = np.random.permutation(n_samples)
            id_rd = np.random.randint(0, n_samples-1)
            x_i = X[id[id_rd], :]
            y_i = y[id[id_rd]]

            y_predicted = np.dot(x_i, self.weights) + self.bias
            error = y_predicted - y_i

            dw=(2/n_samples) * np.sum(x_i * error)
            db=(2/n_samples) * np.sum(error)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            cost = (error**2).sum() / n_samples
            self.costs.append(cost)

            if (epoch+1)%1000==0:
                print(f'epoch: {epoch+1}, weights = {self.weights}, bias = {self.bias}, cost = {cost}')