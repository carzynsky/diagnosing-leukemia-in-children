import pandas as pd
import numpy as np
import math
# from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
# from sklearn.neural_network import MLPClassifierfrom sklearn.model_selection import train_test_splitfrom sklearn.preprocessing import StandardScaler
# from sklearn.metrics import plot_confusion_matrix
# import matplotlib.pyplot as plt

dataPath = 'Data/białaczka.XLS'
resultPath ='Data/chi2.txt'

X = pd.DataFrame(pd.read_excel(dataPath).fillna(0)).values
# y=[]
# xls has been modified (first column class name)
# currentClass=1

# for i in range (len(X)):
#     if X[i][0] != 0:
#         currentClass = X[i][0]
#     y.append(int(currentClass)) 


y=[]
for i in range(len(X)):
    y.append(X[i][0])
X=X[:,2:22]

# chi2

# temp comment
# c, p = chi2(X, y)

# file = open(resultPath, "w")
# for i in range(len(c)):
#     file.write(f'Cecha: {i+1} \t Chi2 = {c[i]} \t P = {p[i]} \n')
# file.close()

# X_new = SelectKBest(chi2, k='all').fit_transform(X,y)
# print(X_new)
# print(X_new.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# mlp
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(256), random_state=1).fit(X, y)
y_pred = clf.predict(X)
# print(clf.predict(X))
# print(clf.score())