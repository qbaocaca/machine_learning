import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
df=pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

sc= StandardScaler()
df['bmi'] = sc.fit_transform(df[['bmi']])

X=df.bmi
y=df.target

# plt.plot(X, y, 'ro')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_train_np= X_train.to_numpy()
X_test_np=X_test.to_numpy()
y_train_np= y_train.to_numpy()
y_test_np=y_test.to_numpy()

X_train_np= np.reshape(X_train_np, (309,1))
X_test_np= np.reshape(X_test_np, (133,1))

X_train = torch.from_numpy(X_train_np.astype(np.float32))
X_test = torch.from_numpy(X_test_np.astype(np.float32))
y_train = torch.from_numpy(y_train_np.astype(np.float32))
y_test = torch.from_numpy(y_test_np.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

model = nn.Linear(in_features=1, out_features=1)

learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

num_epochs = 20000
for epoch in range(num_epochs):
  y_predicted = model(X_train)
  loss = criterion(y_predicted, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (epoch+1)%1000==0:
    print(f'epoch {epoch+1}, loss={loss.item():.4f}')

predicted = model(X_train).detach().numpy()
plt.figure(figsize=(10,7))
plt.plot(X_train, y_train_np, 'ro') # points
plt.plot(X_train, predicted, 'b') # line
plt.show()
