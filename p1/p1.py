import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, ShuffleSplit
import rdkit.Chem

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# store gap values
Y_train = df_train.gap.values
# row where testing examples start
test_idx = df_train.shape[0]
# delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
# delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

# DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
df_all.head()

# Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values

# Retrieving train and test post feature-engineering 
X_train = vals[:test_idx]
X_test = vals[test_idx:]

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

def RMSE (y_hats, y):
	n = len(y)
	total = 0
	for i in xrange(n):
		total += (y_hats[i] - y[i])**2
	return np.sqrt(total/n)

def kFoldCrossVal(k, X1, y, X2):
	kf = KFold(n_splits = k)
	LR = LinearRegression()
	bestValidation = float('inf')
	bestPred = []
	for test, train in kf.split(X1):
		X_tr, X_te, y_tr, y_te = X1[train], X1[test], y[train], y[test]
		LR.fit(X_tr, y_tr)
		LR_pred = LR.predict(X_te)
		valError = RMSE(LR_pred,y_te)
		if valError < bestValidation:
			bestValidation = valError
			bestPred = LR.predict(X2)
	write_to_file("kFold.csv", bestPred)

def bagging(i, p, X1, y, X2):
	ss = ShuffleSplit(n_splits = i, test_size = p)
	LR = LinearRegression()
	predictions = None
	for train, test in ss.split(X1):
		X_tr, X_te, y_tr, y_te = X1[train], X1[test], y[train], y[test]
		LR.fit(X_tr, y_tr)
		LR_pred = LR.predict(X2)
		if predictions is None:
			predictions = LR_pred
		else: predictions = np.add(predictions, LR_pred)
	predictions = np.divide(predictions, i)
	write_to_file("bagging.csv", predictions)

