# Ensemble method of feed forward neural network and gradient boost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Read CSV file
df = pd.read_csv('regress.csv')

# Split dataset into input and output variables

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y),1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize input variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train feed forward neural network model
model_ffnn = MLPRegressor(hidden_layer_sizes=(64, 128, 256, 128), max_iter=1000, random_state=42)
model_ffnn.fit(X_train, y_train)

# Train gradient boost model

joblib.dump(model_ffnn, "FFN.sav")
