import numpy as np
from softmax_regression import SoftmaxRegression

class softmax_SGD(SoftmaxRegression):

    def __init__(self, learning_rate, n_iters, num_classes):
        super().__init__(learning_rate, n_iters, num_classes)

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = dw = np.random.random((self.num_classes, n_features))
        self.bias = db = np.zeros((self.num_classes, 1))

        self.w_cache = []
        self.b_cache = []

        id = np.random.permutation(n_samples)

        Y = np.eye(self.num_classes)[y]

        for epoch in range(n_samples):
            x_i = X[id[epoch], :].reshape((n_features, 1))
            y_i = Y[id[epoch], :].reshape((self.num_classes, 1))

            linear_product = self.get_linear_product(x_i)
            y_predicted = self.softmax_stable(linear_product)
            dw, db = self.compute_gradient(x_i, y_i, y_predicted)
            loss = self.compute_loss_stable(y_i, linear_product)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            self.costs.append(loss)
            self.w_cache.append(self.weights)
            self.b_cache.append(self.bias)

            print(f'epoch: {epoch+1}')
            print(f'weights = {self.weights}')
            print(f'bias = {self.bias}')
            print(f'cost = {loss}')
            print("")
