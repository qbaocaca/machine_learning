import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df= pd.read_csv("C:/Users/DELL/OneDrive/Desktop/caltech_course/machine_learning/fish_classification.csv")

le = LabelEncoder()
df.Species = le.fit_transform(df.Species)

X=df.drop(['Species', 'Weight', 'Length1', 'Length2', 'Length3'], axis=1)
y= df.Species

std = StandardScaler()
X = std.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = Perceptron(eta0=0.01, max_iter=20, random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
