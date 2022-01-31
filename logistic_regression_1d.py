import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression, accuracy
from logistic_regression import compute_loss_numerically_stable
from logistic_regression import visualize_data1d
from logistic_regression import visualize_cost
from logistic_regression import visualize_sigmoid
from logistic_regression import predict_avalue, predict_proba_
from logistic_regression import create_test_case
from logistic_regression import get_different_idx, compute_loss

X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])

y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# visualize_data1d(X,y)

X= X.reshape((X.shape[0], 1))
y= y.reshape((y.shape[0], 1))

model = LogisticRegression(learning_rate=0.01, n_iters=10000)
model.fit(X, y)

# visualize_cost(model)
# visualize_sigmoid(X, y, model)

# Test for one data point

# print(f'study 5.2 hours will {predict_avalue(model, 5.2)}')
# print(f'study 1.5 hours will {predict_avalue(model, 1.5)}')
# print(f'study 3.2 hours will have a {predict_proba_(model, 3.2):.1f}% of pass')
# print(f'study 2.5 hours will have a {predict_proba_(model, 2.5):.1f}% of pass')

# Test for an array of data points

X_test, y_test = create_test_case(200)
visualize_data1d(X_test, y_test)
X_test = X_test.reshape(X_test.shape[0],1)
predictions = model.predict(X_test)
print(f'Test accuracy: {accuracy(y_test, predictions)* 100}%')

# See the misclassified data points

temp = get_different_idx(y_test, predictions)
print('Students study at least 3 hours to get passed!')
print('The misclassified data points:')
for idx in temp:
    print(X_test[idx][0])

visualize_sigmoid(X_test, y_test, model)

print(f'LR classification loss: {compute_loss(X_test, y_test, model)}')