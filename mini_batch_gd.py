import numpy as np
from linear_regression import LinearRegression

class mini_batch_gradient_descend(LinearRegression):

    def __init__(self, learning_rate, n_iters, batch_size):
        super().__init__(learning_rate, n_iters)
        self.batch_size = batch_size

    def create_batch(self, X, y, batch_size):
        n_samples, n_features = X.shape
        id = np.random.permutation(n_samples)
        X_batch = []
        y_batch = []
        for i in range(batch_size):
            id_rd = np.random.randint(0, n_samples-1)
            x_i = X[id[id_rd], :]
            X_batch.append(x_i)
            y_batch.append(y[id[id_rd]])

        return (np.array(X_batch), np.array(y_batch))


    def fit(self, X, y):
        n_samples , n_features = X.shape
        self.weights = dw = np.zeros(n_features)
        self.bias = db = 0.0
        total_loss = 0.0

        for epoch in range(self.n_iters):
            X_batch, y_batch = self.create_batch(X, y, self.batch_size)
            for idx, x_i in enumerate(X_batch):
                y_predicted = np.dot(x_i, self.weights) + self.bias
                error = y_predicted - y_batch[idx]
                dw_i = x_i * error
                db_i = error
                loss = error**2

                dw += dw_i
                db += db_i
                total_loss += loss

            dw = (2/self.batch_size) * dw
            db = (2/self.batch_size) * db
            total_loss = (1/self.batch_size) * total_loss

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.costs.append(total_loss)

            if (epoch+1)%1000==0:
                print(f'epoch: {epoch+1}, '
                f'weights = {self.weights[0]}, '
                f'bias = {self.bias}, '
                f'cost = {total_loss}')
