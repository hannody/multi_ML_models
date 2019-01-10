import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import time
t = time.time()
import sys
if sys.platform == 'darwin':
    train_path = "/Users/mqa/Desktop/Dev/ML/introduction_to_ml_with_python/Data sets/superconduct/train.csv"
else:
    train_path = "/home/axis/Desktop/ml_work/intro_to_ml/Data sets/superconduct/train.csv"


X = pd.read_csv(train_path)

y = X["critical_temp"]

X = X.drop(["critical_temp"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


lr = LinearRegression(normalize=True).fit(X_train, y_train)

ridge = Ridge().fit(X_train, y_train)

lasso = Lasso(alpha=35).fit(X_train, y_train)

tree_reg = DecisionTreeRegressor().fit(X_train, y_train)

forest = RandomForestRegressor(n_estimators=100, random_state=0,
                               n_jobs=-1).fit(X_train, y_train)  # -1 means all availabe cores

gbrt = GradientBoostingRegressor(
    random_state=0, max_depth=15).fit(X_train, y_train)

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

mlpr = MLPRegressor(max_iter=1000, alpha=1, random_state=0, activation='relu').fit(
    X_train_scaled, y_train)

print("The score of LR:{:.2f}".format(lr.score(X_test, y_test)))

print("The score of ridge:{:.2f}".format(
    ridge.score(X_test, y_test)))  # pylint: disable=no-member

print("The score of lasso:{:.2f}".format(lasso.score(X_test, y_test)))

print("The score of DecisionTreeRegressor:{:.2f}".format(
    tree_reg.score(X_test, y_test)))

print("The score of RandomForestRegressor:{:.2f}".format(
    forest.score(X_test, y_test)))

print("The score of GradientBoostingRegressor:{:.2f}".format(
    gbrt.score(X_test, y_test)))

print(
    "The score of Nural Network/MLPRegressor:{:.2f}".format(mlpr.score(X_test_scaled, y_test)))

print("Time taken is :{:.2f}".format((time.time() - t)/60))
