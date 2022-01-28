import numpy as np
from Perceptron import Perceptron

class perceptron_SGD (Perceptron):

    def __init__(self, learning_rate, n_iters):
        super().__init__(learning_rate, n_iters)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.dot(np.linalg.pinv(X), y)
        self.bias = 0

        y_ = np.array([1 if i>0 else 0 for i in y])
        for epoch in range(self.n_iters):
            id = np.random.permutation(n_samples)
            id_rd = np.random.randint(0, n_samples-1)
            x_i = X[id[id_rd], :]
            y_i = y_[id[id_rd]]

            linear_product = np.dot(x_i, self.weights) + self.bias
            y_predicted = self.activation_func(linear_product)
            error = y_i - y_predicted

            self.weights += self.lr * error * x_i
            self.bias+= self.lr * error
            self.costs.append(int(error!= 0.0))

            if (epoch+1)%100 == 0:
                print(f'epoch: {epoch+1}, weights = {self.weights}, bias = {self.bias}')
