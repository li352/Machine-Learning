#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 22:09:19 2019

@author: linduri
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
dataset
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values
y

#Take care of missing dtaa
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer=imputer.fit(X[:, 1:3])
imputer
X[:, 1:3]=imputer.transform(X[:, 1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
labelencoder=labelencoder.fit(X[:,0])
X[:,0]=labelencoder.transform(X[:,0])
X

from sklearn.preprocessing import OneHotEncoder
Onehotencoder_X=OneHotEncoder(categorical_features=[0])
X=Onehotencoder_X.fit_transform(X).toarray()
X

labelencoder_y=LabelEncoder()
labelencoder_y=labelencoder.fit(y)
y=labelencoder_y.transform(y)
y


# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X.shape
y.shape


# Feature Scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train[:,3:6]=ss.fit_transform(X_train[:,3:6])
#X_train[:,3:4]=ss.fit_transform(X_train[:,3:4])
X_train
X_test[:,3:5]=ss.transform(X_test[:,3:5])

