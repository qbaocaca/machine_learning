import numpy as np
from Perceptron import Perceptron

class LogisticRegression(Perceptron):

    def __init__(self, learning_rate, n_iters):
        super().__init__(learning_rate, n_iters)
        self.activation_func = self.sigmoid

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_product = self.get_linear_product(x_i)
                y_predicted = self.activation_func(linear_product)

                error = y_predicted - y[idx]

                dw = (2/n_samples) * np.dot(x_i.T, error)
                db = (2/n_samples) * np.sum(error)

                self.weights -= self.lr * dw # minus sign bc y_predicted - y[idx]
                self.bias -= self.lr * db # otherwise, plus sign

            linear_output = self.get_linear_product(X)
            cost = compute_loss_numerically_stable(y, linear_output)
            self.costs.append(cost[0][0])

            if (epoch+1)%1000==0:
                print(f'epoch: {epoch+1}, weights = {self.weights[0]}, bias = {self.bias}, cost = {cost[0][0]}')

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def predict(self, X):
        y_predicted = self.activation_func(self.get_linear_product(X))
        y_predicted_class = [1 if i >= 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def predict_proba_(self, X):
        return self.activation_func(self.get_linear_product(X))

    def get_linear_product(self, X):
        return np.dot(X, self.weights) + self.bias

def compute_loss_numerically_stable(y, z):
    loss_1 = np.log(1 + np.exp(-z))
    loss_0 = z + np.log(1 + np.exp(-z))
    loss_1 = loss_1.reshape((y.shape[1], y.shape[0]))
    loss_0 = loss_0.reshape((y.shape[1], y.shape[0]))
    return np.dot(loss_1, y) + np.dot(loss_0, 1-y)