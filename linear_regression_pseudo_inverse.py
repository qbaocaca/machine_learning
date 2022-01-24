import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

x0 = np.ones(X_train_np.shape[0])
x0 = x0.reshape(x0.shape[0], 1)

X_final = np.concatenate([X_train_np, x0], axis=1)

w = np.dot(np.linalg.pinv(X_final), y_train_np)

fig, ax = plt.subplots()

plt.title('Diabetes predictions')
plt.xlabel('bmi in std')
plt.ylabel('Score')

ax.scatter(X_train_np, y_train_np)

x_min, x_max = ax.get_xlim()
y_min, y_max = w[1]*x_min + w[0], w[1]*x_max + w[0]

ax.set_xlim([x_min, x_max])
_ = ax.plot([x_min, x_max], [y_min, y_max])
plt.show()
