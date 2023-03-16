import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import joblib


dataset=pd.read_csv('data/regress.csv')


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 0)



sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(Xtrain)
y_train = sc_y.fit_transform(ytrain)


regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

s=sc_X.transform(Xtest)

a=regressor.predict(s).reshape(-1,1)

y_pred = sc_y.inverse_transform(a)


r2_score(ytest, y_pred)

print(mean_absolute_error(ytest, y_pred))
print(r2_score(ytest,y_pred))

joblib.dump(regressor, "rf_model.sav")

