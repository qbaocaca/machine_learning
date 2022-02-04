import numpy as np
from Perceptron import Perceptron

class SoftmaxRegression(Perceptron):

    def __init__(self, learning_rate, n_iters, num_classes):
        super().__init__(learning_rate, n_iters)
        self.w_cache = None
        self.b_cache = None
        self.num_classes = num_classes

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = dw = np.zeros((self.num_classes, n_features))
        self.bias = db = np.zeros((self.num_classes, 1))
        total_loss = 0.0
        self.w_cache = []
        self.b_cache = []

        Y = np.eye(self.num_classes)[y]

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                x_i = X[idx, :].reshape((n_features, 1))
                y_i = Y[idx, :].reshape((self.num_classes, 1))

                linear_product = self.get_linear_product(x_i)
                y_predicted = self.softmax_stable(linear_product)
                dw_i, db_i = self.compute_gradient(x_i, y_i, y_predicted)
                loss = self.compute_loss_stable(y_i, linear_product)

                dw += dw_i
                db += db_i
                total_loss += loss

            dw = (1.0/n_samples) * dw
            db = (1.0/n_samples) * db
            total_loss = (1.0/n_samples) * total_loss

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            self.costs.append(total_loss)
            self.w_cache.append(self.weights)
            self.b_cache.append(self.bias)

            if (epoch+1)%100==0:
                print(f'epoch: {epoch+1}')
                print(f'weights = {self.weights}')
                print(f'bias = {self.bias}')
                print(f'cost = {total_loss}')
                print("")

    def predict(self, X):
        result = []
        n_samples, n_features = X.shape
        for idx, x_i in enumerate(X):
            x_i = X[idx, :].reshape((n_features, 1))
            linear_product = self.get_linear_product(x_i)
            y_predicted = self.softmax_stable(linear_product)
            max_val = y_predicted.max()
            temp = y_predicted.tolist()
            max_idx = temp.index(max_val)
            result.append(max_idx)
        return np.array(result)

    def get_linear_product(self, X):
        return np.matmul(self.weights, X) + self.bias

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def softmax_stable(self, z):
        return np.exp(z - z.max()) / np.sum(np.exp(z - z.max()))

    def compute_loss(self, y, a):
        return -1.0 * np.sum(y * np.log(a))

    def compute_loss_stable(self, y, z):
        return -1.0 * np.sum(y * (z - z.max() - np.log(np.sum(np.exp(z - z.max())))))

    def compute_gradient(self, X, y, a):
        # dL / da
        # dl_da = -y / a
        dl_da = self.cal_dl_da(y, a)

        # da / dz
        # compute am
        extension = np.ones((1, self.num_classes))  # (1, 3)
        am = np.matmul(a, extension)           # (3, 1) x (1, 3)
        one = np.identity(self.num_classes)         # (3, 3)
        da_dz = am * (one - am.T)              # (3, 3)

        # dl / dz
        dl_dz = np.matmul(da_dz, dl_da)        # (3, 1)

        # dl / dw
        dl_dw = np.matmul(dl_dz, X.T)          # (3, 2)

        # dl / db
        dl_db = dl_dz.copy()                   # (3, 1)

        return dl_dw, dl_db

    def cal_dl_da (self, y, a):
        w = np.zeros(a.shape)
        for i in range(a.shape[0]):
            if y[i][0] == 0:
                w[i][0] = 0
            elif a[i][0] > 1.7976e-10: # shouldn't have this!
                w[i][0] = -y[i][0]
            else:
                w[i][0] = -y[i][0] / a[i][0]
        return w