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

X_final = np.concatenate([x0, X_train_np], axis=1)

def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = X_final[true_i, :]
    yi = y_train_np[true_i]
    a = np.dot(xi, w) - yi
    return xi*a

def SGD(w_init, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X_final.shape[0]
    count = 0
    for it in range(10):
        # shuffle data
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check)/len(w) < 1e-3:
                    return w
                w_last_check = w_this_check
    return w

weights = SGD(0, 0.01)
last_weight = weights[-1]

print(weights)
print(f'last_weight={last_weight}')

w = last_weight[1] # w roughly 47
b = last_weight[0] # b roughly 145
# actually the same as what i implemented!

fig, ax = plt.subplots()

plt.title('Diabetes predictions')
plt.xlabel('bmi in std')
plt.ylabel('Score')

ax.scatter(X_train_np, y_train_np)

x_min, x_max = ax.get_xlim()
y_min, y_max = w*x_min + b, w*x_max + b

ax.set_xlim([x_min, x_max])
_ = ax.plot([x_min, x_max], [y_min, y_max], c='r')
plt.show()
