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
        self.weights = dw = np.zeros(n_features)
        # self.weights = pseudo_inverse (X, y)
        self.bias = db = 0.0
        total_loss = 0.0

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_predicted = np.dot(x_i, self.weights) + self.bias
                error = y_predicted - y[idx]
                dw_i = x_i * error
                db_i = error
                loss = error**2

                dw += dw_i
                db += db_i
                total_loss += loss

            dw = (2/n_samples) * dw
            db = (2/n_samples) * db
            total_loss = (1/n_samples) * total_loss

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.costs.append(total_loss)

            if (epoch+1)%1000==0:
                print(f'epoch: {epoch+1}, '
                f'weights = {self.weights[0]:.4f}, '
                f'bias = {self.bias:.4f}, '
                f'cost = {total_loss}')

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def mean_squared_error(y_true, y_pred):
    error = y_pred - y_true
    return (error**2).sum() / len(y_true)

# shouldn't be right!
# def pseudo_inverse (X, y):
#     temp = np.dot(X, X.T)
#     temp = np.array(temp)
#     temp = temp.reshape((1,1))
#     temp2 = np.linalg.inv(temp)
#     temp3 = np.dot(temp2, X)
#     return np.dot(y, temp3)