import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import joblib


dataset=pd.read_csv('class.csv')
data=pd.read_csv('class.csv')
#seperating independent and dependent variables

x = data.drop(['Movement'], axis=1)
y = data['Movement']

from sklearn.model_selection import train_test_split
train_x,valid_x,train_y,valid_y = train_test_split(x,y, random_state = 101, stratify=y)

model1 = LogisticRegression()
model1.fit(train_x,train_y)




joblib.dump(model1, "rf_model_class.sav")

