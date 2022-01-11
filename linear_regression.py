import numpy as np

class LinearRegression:

    def __init__(self, learning_rate, n_iters):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.costs = []

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_predicted = np.dot(x_i, self.weights) + self.bias
                error = y_predicted - y[idx]
                dw=(2/n_samples) * np.sum(x_i * error)
                db=(2/n_samples) * np.sum(error)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                cost = (error**2).sum() / n_samples
                self.costs.append(cost)

            if (epoch+1)%1000==0:
                print(f'epoch: {epoch+1}, weights = {self.weights}, bias = {self.bias}, cost = {cost}')


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def mean_squared_error(y_true, y_pred):
    error = y_pred - y_true
    return (error**2).sum() / len(y_true)