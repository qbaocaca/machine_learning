import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression, compute_loss_numerically_stable

X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])

y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# Visualization
def visualize_data (X,y):
    colormap = np.array(['r', 'b'])
    plt.ylim(bottom=-0.1, top=1.1)
    plt.scatter(X,y, c=colormap[y])
    plt.show()

# visualize_data(X,y)

X= X.reshape((X.shape[0], 1))
y= y.reshape((y.shape[0], 1))

model = LogisticRegression(learning_rate=0.01, n_iters=10000)
model.fit(X, y)

# Visual Cost function
def visualize_cost (model):
    plt.figure(figsize=(10,7))
    plt.grid()
    plt.plot(np.array(model.costs))
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Gradient Descent')
    plt.show()

# visualize_cost(model)

# Visual sigmoid curve after train

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

# visualize_sigmoid(X, y, model)

# Test for one data point

def predict_avalue (model, value):
    if model.predict(value)[0] == 1:
        return 'pass'
    else:
        return 'fail'

def predict_proba_(model, value):
    temp = model.predict_proba_(value)
    return round(temp[0]*100, 1)

print(f'study 5.2 hours will {predict_avalue(model, 5.2)}')
print(f'study 1.5 hours will {predict_avalue(model, 1.5)}')
print(f'study 3.2 hours will have a {predict_proba_(model, 3.2)}% of pass')
print(f'study 2.5 hours will have a {predict_proba_(model, 2.5)}% of pass')

# Test for an array of data points

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def create_test_case(num):
    X = []
    for _ in range(num):
        X.append(np.random.uniform(0,5))

    y = [1 if i >= 3 else 0 for i in X] # study at least 3 hours to pass

    return np.array(X), np.array(y)

X_test, y_test = create_test_case(200)
# visualize_data(X_test, y_test)
# X_test = X_test.reshape(X_test.shape[0],1)
# predictions = model.predict(X_test)
# print(f'Test accuracy: {accuracy(y_test, predictions)* 100}%')

def get_different_idx(y_true, y_pred):
    result = []
    for i in range(y_true.shape[0]):
        if y_true[i] != y_pred[i]:
            result.append(i)
    return result

# temp = get_different_idx(y_test, predictions)
# print('Students study at least 3 hours to get passed!')
# print('The misclassified data points:')
# for idx in temp:
#     print(X_test[idx][0])

# visualize_sigmoid(X_test, y_test, model)

def compute_loss(X, y, model):
    X = X.reshape(X.shape[0],1)
    # y = y.reshape(y.shape[0],1)
    linear_product = model.get_linear_product(X)
    temp = compute_loss_numerically_stable(y, linear_product)
    return round(temp[0][0], 1)

print(f'LR classification loss: {compute_loss(X_test, y_test, model)}')