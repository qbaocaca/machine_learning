import numpy as np
from Perceptron import Perceptron
import matplotlib.pyplot as plt

class LogisticRegression(Perceptron):

    def __init__(self, learning_rate, n_iters):
        super().__init__(learning_rate, n_iters)
        self.activation_func = self.sigmoid
        # self.w_cache = None
        # self.b_cache = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = dw = np.zeros(n_features)
        self.bias = db = 0.0
        total_loss = 0.0
        # self.w_cache = []
        # self.b_cache = []

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_product = self.get_linear_product(x_i)
                y_predicted = self.activation_func(linear_product)

                error = y_predicted - y[idx]

                dw_i = x_i * error
                db_i = error
                loss = compute_loss_numerically_stable(y[idx], linear_product)

                dw += dw_i
                db += db_i
                total_loss += loss

            dw = (2/n_samples) * dw
            db = (2/n_samples) * db
            total_loss = (1/n_samples) * total_loss
            self.weights -= self.lr * dw # minus sign bc y_predicted - y[idx]
            self.bias -= self.lr * db # otherwise, plus sign

            self.costs.append(total_loss)
            # self.w_cache.append(self.weights)
            # self.b_cache.append(self.bias)

            # 2d
            # if (epoch+1)%1000==0:
            #     print(f'epoch: {epoch+1}, weights = {self.weights[0]}, {self.weights[1]}, '
            #     f'bias = {self.bias}, cost = {total_loss}')

            # 1d
            if (epoch+1)%1000==0:
                print(f'epoch: {epoch+1}, weights = {self.weights[0]}, '
                f'bias = {self.bias[0]}, cost = {total_loss[0]}')

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

# shouldn't be right!
# def compute_loss_numerically_stable(y, z):
#     loss_1 = np.log(1 + np.exp(-z))
#     loss_0 = z + np.log(1 + np.exp(-z))
#     loss_1 = loss_1.reshape((y.shape[1], y.shape[0]))
#     loss_0 = loss_0.reshape((y.shape[1], y.shape[0]))
#     return np.dot(loss_1, y) + np.dot(loss_0, 1-y)

def compute_loss_numerically_stable(y, z):
    return -1 * (-y * np.log(1 + np.exp(-z)) +
                (1-y) * (-z - np.log(1+np.exp(-z))))

def create_data(num, xx, xy, yx, yy, c1, c2):
    X = []
    for i in range(num):
        temp = []
        x_ = np.random.uniform(xx,xy)
        y_ = np.random.uniform(yx,yy)
        # y_ = np.random.random()
        temp.append(x_)
        temp.append(y_)
        y_i = np.random.randint(c1,c2)
        temp.append(y_i)
        X.append(temp)
    return np.array(X)

def visualize_data (X, y):
    # colormap = np.array(['r', 'c'])
    colormap = np.array(['m', 'c', 'y'])
    # plt.ylim(bottom=-0.1, top=1.1)
    plt.scatter(X[:, 0], X[:, 1], c=colormap[y])
    plt.show()

def visualize_data1d (X,y):
    colormap = np.array(['r', 'b'])
    plt.ylim(bottom=-0.1, top=1.1)
    plt.scatter(X,y, c=colormap[y])
    plt.show()

def visualize_cost (model):
    plt.figure(figsize=(10,7))
    plt.grid()
    plt.plot(np.array(model.costs))
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Gradient Descent')
    plt.show()

def compute_loss(X, y, model):
    temp1 = compute_loss_numerically_stable(y, model.get_linear_product(X))
    sum=0.0
    for i in temp1:
        sum+=i
    return (1/X.shape[0]) * sum

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def visualize_sigmoid(X, y, model):

    X= X.reshape((X.shape[0]))
    y= y.reshape((y.shape[0]))

    # plt.figure(figsize=(10,7))
    colormap2 = np.array(['r', 'b'])
    plt.scatter(X,y, c= colormap2[y])
    xx = np.linspace(0, 6, 1000)
    w0 = model.bias
    w1 = model.weights
    threshold = -w0/w1
    yy = model.sigmoid(w0 + w1*xx)
    # plt.axis([8, 25.5, -0.5, 1.5])
    plt.ylim(bottom=-0.1, top=1.1)
    plt.plot(xx, yy, 'black', linewidth = 2)
    plt.plot(threshold, .5, 'g^', markersize = 15)
    plt.xlabel('studying hours')
    plt.ylabel('predicted probability of pass')
    plt.show()

def predict_avalue (model, value):
    if model.predict(value)[0] == 1:
        return 'pass'
    else:
        return 'fail'

def predict_proba_(model, value):
    temp = model.predict_proba_(value)
    return temp[0]*100

def create_test_case(num):
    X = []
    for _ in range(num):
        X.append(np.random.uniform(0,5))

    y = [1 if i >= 3 else 0 for i in X] # study at least 3 hours to pass

    return np.array(X), np.array(y)

def get_different_idx(y_true, y_pred):
    result = []
    for i in range(y_true.shape[0]):
        if y_true[i] != y_pred[i]:
            result.append(i)
    return result

# fail version!
# def compute_loss(X, y, model):
#     X = X.reshape(X.shape[0],1)
#     # y = y.reshape(y.shape[0],1)
#     linear_product = model.get_linear_product(X)
#     temp = compute_loss_numerically_stable(y, linear_product)
#     return round(temp[0][0], 1)