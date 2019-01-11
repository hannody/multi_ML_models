from sklearn.neural_network import MLPRegressor
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import time

t = time.time()
if sys.platform == 'darwin':
    train_path = "/Users/mqa/Desktop/Dev/ML/introduction_to_ml_with_python/Data sets/superconduct/train.csv"
else:
    train_path = "/home/axis/Desktop/ml_work/intro_to_ml/Data sets/superconduct/train.csv"


X = pd.read_csv(train_path)

y = X["critical_temp"]

X = X.drop(["critical_temp"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# "Scaling the data for MLPR, without scaling the score is around %60"

# Compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)

# Compute the STD per feature on the training set
std_on_train = X_train.std(axis=0)

# subtract the mean, and scale by inverse STD
# afterward, mean = 0 and std =1
X_train_scaled = (X_train - mean_on_train)/std_on_train


# Scaling the test set(the same transformation on the test set)
X_test_scaled = ((X_test - mean_on_train)/std_on_train)

mlpr = MLPRegressor(max_iter=400, alpha=0.001, random_state=0, activation='tanh', solver='lbfgs').fit(
    X_train_scaled, y_train)

print(
    "The score of Nural Network/MLPRegressor-tanh:{:.2f}".format(mlpr.score(X_test_scaled, y_test)))


print("Time taken is :{:.2f}".format((time.time() - t)/60))
