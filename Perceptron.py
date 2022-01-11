import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):

        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self.unit_step_func

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples ,n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i>0 else 0 for i in y]) # in case y is -1 or 1

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_pred)
                self.weights += update*x_i
                self.bias+= update

            if (epoch+1)%100 == 0:
                print(f'epoch: {epoch+1}, weights = {self.weights}, bias = {self.bias}')

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)


    def unit_step_func(self, X):
        return np.where(X>=0, 1, 0)
