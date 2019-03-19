#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:18:31 2019

@author: srinivasa.vemula
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


seed = 9 
# Load the data
dataset = pd.read_csv("data.csv")
# Drop rows having empty values
dataset = dataset.dropna()

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

#Feature scaling
scaler_X= StandardScaler()
X = scaler_X.fit_transform(X)

# Preprocess Y
ohe = OneHotEncoder()
Y = Y.reshape(-1,1)
Y = ohe.fit_transform(Y).toarray()

def baseline_model():
    # create the model
    model = Sequential()
    model.add(Dense(13, input_dim=13, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=5, verbose=1)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))