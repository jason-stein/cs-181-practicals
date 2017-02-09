import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV
import rdkit.Chem

# importing old for gap values
df_train = pd.read_csv("train.csv")
# df_test = pd.read_csv("test.csv")
df_fpts = pd.read_csv("pruned512.csv")

df_fpts = df_fpts.drop(df_fpts.columns[0], axis=1)
df_fpts.columns = ['fpts_'+str(i+1) for i in range(df_fpts.shape[1])]

# store gap values
Y_train = df_train.gap.values
# row where testing examples start
test_idx = df_train.shape[0]
# delete 'Id' column
# df_test = df_test.drop(['Id'], axis=1)
# delete 'gap' column
# df_train = df_train.drop(['gap'], axis=1)

# DataFrame with all train and test examples so we can more easily apply feature engineering on
# df_all = pd.concat((df_train, df_test), axis=0)
# df_all.head()

# df_all = df_all.join(df_fpts)

# Drop the 'smiles' column
# df_all = df_all.drop(['smiles'], axis=1)
df_all = df_fpts

# USED FOR GETTING RID OF ALL 0 COLUMNS
cols_to_drop = (df_all.columns[(df_all == 0).all()]).values
df_all = df_all.drop(cols_to_drop, axis=1)
# USED FOR GETTING RID OF ALL 0 COLUMNS

# names = list(df_all.columns.values)
# feature_importances = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)
# df_all = p1.df_all.drop([j for (i,j) in feature_importances if i < 0.0001], axis=1)

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
	LR = RandomForestRegressor()
	bestValidation = float('inf')
	bestPred = []

	param_grid = {"n_estimators": [5, 10, 15],
              "max_features": ["log2","sqrt","auto"]}

	for test, train in kf.split(X1):
		X_tr, X_te, y_tr, y_te = X1[train], X1[test], y[train], y[test]
		GS = GridSearchCV(LR, param_grid = param_grid)
		GS.fit(X_tr,y_tr)
		LR_pred = GS.predict(X_te)
		valError = RMSE(LR_pred,y_te)
		print valError
		if valError < bestValidation:
			bestValidation = valError
			bestPred = LR.predict(X2)
		print bestValidation

	write_to_file("512RFkFold.csv", bestPred)

def bagging(i, p, X1, y, X2):
	print "Bagging"
	ss = ShuffleSplit(n_splits = i, test_size = p)
	print "Split"
	LR = LinearRegression()
	predictions = None
	j = 0
	for train, test in ss.split(X1):
		print j
		X_tr, X_te, y_tr, y_te = X1[train], X1[test], y[train], y[test]
		LR.fit(X_tr, y_tr)
		LR_pred = LR.predict(X2)
		if predictions is None:
			predictions = LR_pred
		else: predictions = np.add(predictions, LR_pred)
		j += 1
	predictions = np.divide(predictions, i)
	write_to_file("LRbagging.csv", predictions)

def gridSearchRF(X1, y, X2):
	RF = RandomForestRegressor()
	param_grid = {"n_estimators": [5, 10, 15], "max_features": ["log2","sqrt","auto"]}
	GS = GridSearchCV(RF, param_grid = param_grid)
	GS.fit(X1,y)
	RF_pred = GS.predict(X2)
	print GS.best_params
	print RMSE(RF_pred, y)
	write_to_file("gridSearchRF.csv", RF_pred)




