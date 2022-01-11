import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Perceptron import Perceptron
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib.colors import ListedColormap

df= pd.read_csv("C:/Users/DELL/OneDrive/Desktop/caltech_course/machine_learning/fish_classification.csv")

le = LabelEncoder()
df.Species = le.fit_transform(df.Species)

X=df.drop(['Species', 'Weight', 'Length1', 'Length2', 'Length3'], axis=1)
y= df.Species

std = StandardScaler()
X = std.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Visualization
# idx_bream = y_train == 0
# idx_perch = y_train == 1

# plt.figure(figsize=(10,7))
# plt.scatter(X_train[idx_bream, 0], X_train[idx_bream, 1], color='red',
# marker='s', label="Bream")
# plt.scatter(X_train[idx_perch, 0], X_train[idx_perch, 1],
# color='blue', marker='x', label="Perch")
# plt.xlabel('Height')
# plt.ylabel('Width')
# plt.legend(loc='upper left')
# plt.show()

model=Perceptron(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# labels = [0, 1]
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(labels))
# plt.xticks(tick_marks, labels)
# plt.yticks(tick_marks, labels)

# sns.heatmap(pd.DataFrame(cm), annot = True, cmap = "Blues" ,fmt = 'g')
# ax.xaxis.set_label_position("bottom")
# plt.tight_layout()
# plt.title('Confusion Matrix')
# plt.ylabel('Ground-truth')
# plt.xlabel('Prediction')
# plt.show()


# plot_decision_boundary
plt.figure(figsize=(10,7))
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                        np.arange(x2_min, x2_max, 0.02))
Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

for idx, cl in enumerate(np.unique(y_train)):
      plt.scatter(x=X[y == cl, 0],
                  y=X[y == cl, 1],
                  alpha=0.8,
                  c=colors[idx],
                  marker=markers[idx],
                  label=cl,
                  edgecolor='black')

plt.xlabel('Height')
plt.ylabel('Width')
plt.legend(loc='upper left')
plt.show()