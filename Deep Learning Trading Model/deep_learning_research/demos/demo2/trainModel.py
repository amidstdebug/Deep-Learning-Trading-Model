#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import xgboost as xgb
import joblib

trainData_file = 'trainData.csv'
train_data = pd.read_csv(trainData_file, index_col=None).values

model = xgb.XGBRegressor()
model.fit(train_data[:, 1:], train_data[:, 0])
joblib.dump(model, 'model.pkl')
