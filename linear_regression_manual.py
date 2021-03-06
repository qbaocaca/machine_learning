import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_regression import LinearRegression, mean_squared_error
from lr_sgd import lr_SGD
from mini_batch_gd import mini_batch_gradient_descend

diabetes = datasets.load_diabetes()
df=pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

sc= StandardScaler()
df['bmi'] = sc.fit_transform(df[['bmi']])

X=df.bmi
y=df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_train_np= X_train.to_numpy()
X_test_np=X_test.to_numpy()
y_train_np= y_train.to_numpy()
y_test_np=y_test.to_numpy()

X_train_np= np.reshape(X_train_np, (X_train.shape[0],1))
X_test_np= np.reshape(X_test_np, (X_test_np.shape[0],1))

# model = LinearRegression(learning_rate=0.01, n_iters=10000)
model = lr_SGD(learning_rate=0.01, n_iters=0)
# model = mini_batch_gradient_descend(learning_rate=0.01, n_iters=10000, batch_size=20)
model.fit(X_train_np, y_train_np)

predictions = model.predict(X_test_np)
print(f'MSE: {mean_squared_error(y_test_np, predictions)}')

# a line can be drawn here without concatenating a bias coloumn
# is because the predict method alr calculates the y coordinate
# so the line is drawn based on the points not the weights

plt.figure(figsize=(10,7))
plt.scatter(X_test_np, y_test_np, c='steelblue', edgecolor='white', s=70)
plt.plot(X_test_np, predictions, 'r')
plt.show()

plt.figure(figsize=(10,7))
plt.plot(range(1, len(model.costs) + 1), model.costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Gradient Descent')
plt.show()
