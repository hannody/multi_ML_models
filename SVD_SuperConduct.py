import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from 
import time
t = time.time()

train_path = "/Users/mqa/Desktop/Dev/ML/Data sets/superconduct/train.csv"

X  = pd.read_csv(train_path)

y = X["critical_temp"]

X = X.drop(["critical_temp"] , axis =1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


lr = LinearRegression(normalize= True).fit(X_train, y_train)

ridge = Ridge().fit(X_train, y_train)

lasso = Lasso(alpha=35).fit(X_train, y_train)

tree_reg= DecisionTreeRegressor().fit(X_train, y_train)

forest = RandomForestRegressor(n_estimators= 100, random_state= 0, n_jobs= -1).fit(X_train, y_train)# -1 means all availabe cores

gbrt = GradientBoostingRegressor(random_state=0, max_depth=15).fit(X_train, y_train)

print("The score of LR:{:.2f}".format(lr.score(X_test, y_test)))

print("The score of ridge:{:.2f}".format(ridge.score(X_test, y_test)))

print("The score of lasso:{:.2f}".format(lasso.score(X_test, y_test)))

print("The score of DecisionTreeRegressor:{:.2f}".format(tree_reg.score(X_test, y_test)))

print("The score of RandomForestRegressor:{:.2f}".format(forest.score(X_test, y_test)))

print("The score of GradientBoostingRegressor:{:.2f}".format(gbrt.score(X_test, y_test)))

print("Time taken is :{:.2f}".format((time.time() - t)/60))

#import graphviz
#from sklearn.tree import export_graphviz
#from IPython.display import display

#export_graphviz(tree_reg, out_file="tree.dot")


#with open("tree.dot") as f:
#    dot_grapgh = f.read()
#display(graphviz.Source(dot_grapgh))
