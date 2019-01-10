from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
boston_sk_set = load_boston()

#print(boston_sk_set.keys())
p = "*"*100

path_trainset = "/Users/mqa/Desktop/Dev/ML/Data sets/Boston Housing/all/train.csv"
path_test ="/Users/mqa/Desktop/Dev/ML/Data sets/Boston Housing/all/test.csv"

# reading the training data set into a dataframe..
train_df = pd.read_csv(path_trainset, index_col="ID")

# extracting featues and labels for training..
X = train_df.iloc[:, :-1] # training features
y = train_df.iloc[:, len(train_df.columns)-1] # training labels




y_train = train_df.medv

X_train = train_df.drop(["medv"], axis = 1)

# building and fitting a liner regression model..
lr = LinearRegression().fit(X_train, y_train)

# reading the testing data set as a dataframe
X_test = pd.read_csv(path_test, index_col="ID")
y_test = pd.read_csv("/Users/mqa/Desktop/Dev/ML/Data sets/Boston Housing/all/submission_example.csv", index_col="ID")

print(len(X_test))
print(len(X_train))
print(len(y_test))
print(len(y_train))

print(lr.score(X_test, y_test))


