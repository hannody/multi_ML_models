import time

start_time = time.time()

from sklearn.datasets import load_iris
import numpy as np


iris_dataset = load_iris()


from sklearn.model_selection import train_test_split

import mglearn

X_train, X_test,  y_train, y_test = train_test_split(
    iris_dataset.data, iris_dataset.target, random_state=0)


import pandas as pd

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)

y_pred = knn.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

print("Model Accuracy/Score:  {:.2f}".format(knn.score(X_test, y_test)))

print("Time elapsed in seconds: {:.2f}".format(time.time() - start_time))
