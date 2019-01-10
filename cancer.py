

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer_set = load_breast_cancer()

print('#' * 100, '\n')

print("Cancer Data Keys are:{}".format(cancer_set.keys()))

# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

print('-' * 100, '\n')

X_train, X_test, y_train, y_test = train_test_split(cancer_set.data, cancer_set.target, random_state=0)

print(len(X_train))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 11)

import time

t= time.time()

knn.fit(X_train, y_train)

import numpy as np

X_new = np.random.rand(1,30)

print("The score for K-NN: {:.2f}".format(knn.score(X_test, y_test)))

# Using Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

t= time.time()

log_reg.fit(X_train, y_train )

print("The score of a Logistic Regression model is: {:.2f}".format(log_reg.score(X_test, y_test)))