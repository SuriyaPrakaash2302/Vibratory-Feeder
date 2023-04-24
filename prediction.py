import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import joblib

def predict(data):
	dataset=pd.read_csv('regress.csv')
	FFN = joblib.load("FFN.sav")
	GB = joblib.load("GBoost.sav")
	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values
	y = y.reshape(len(y),1)
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 0)
	sc_X = StandardScaler()
	sc_y = StandardScaler()
	X_train = sc_X.fit_transform(Xtrain)
	y_train = sc_y.fit_transform(ytrain)
	s=sc_X.transform(data)
	a=FFN.predict(s).reshape(-1,1)
	b=GB.predict(s).reshape(-1,1)

	y_pred = (a + b)/2
	return 10/y_pred

def classi(data):
	clf = joblib.load("rf_model_class.sav")
	a = clf.predict(data).reshape(-1,1)
	if (a==1):
		return "Yes"
	else:
		return "No"

def voltager(data):
	df=pd.read_csv('regress.csv')
	XB = joblib.load("XB.sav")
	
	X = df.iloc[:, 2:-1].values
	y = df.iloc[:, 0:2].values
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 0)
	sc_X = StandardScaler()
	sc_y = StandardScaler()
	X_train = sc_X.fit_transform(Xtrain)
	y_train = sc_y.fit_transform(ytrain)
	s=sc_X.transform(data)
	a=XB.predict(s).reshape(-1,2)
	return a
