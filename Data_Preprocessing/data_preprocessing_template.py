# -*- coding: utf-8 -*-
"""
Data Preprocessing
@author: Vanitha

The following code helps to preprocess the data before building the model
"""
#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.NaN,strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
print(x)

 
#Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
LabelEncoder_x=LabelEncoder()
x[:,0]=LabelEncoder_x.fit_transform(x[:,0])
preprocess = ColumnTransformer(transformers=[('onehot', OneHotEncoder(),[0])],remainder="passthrough")
x = preprocess.fit_transform(x)

LabelEncoder_y=LabelEncoder()
y=LabelEncoder_y.fit_transform(y)


#Splitting training test and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
