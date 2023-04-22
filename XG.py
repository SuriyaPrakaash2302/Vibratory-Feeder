from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib


df = pd.read_csv('regress.csv')

# Split dataset into input and output variables
X = df.iloc[:, 2:-1].values
y = df.iloc[:, 0:2].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create an XGBRegressor object
xgb = XGBRegressor(random_state=42)

# Define a dictionary of hyperparameters and their values to try
param_grid = {
    'n_estimators': [50, 40],
    'max_depth': [3],
    'learning_rate': [0.1],
}

# Create a GridSearchCV object with the XGBRegressor and parameter grid
grid_search = GridSearchCV(xgb, param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Make predictions on the test data using the best estimator
y_pred = grid_search.best_estimator_.predict(X_test)

joblib.dump(grid_search, "XB.sav")
