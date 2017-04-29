# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import csv

train = pd.read_csv("target/target_train.csv")
test = pd.read_csv("result.csv", header = None)

# train = train[train['date'] <= "2014-11-30"]
# print train

#=============================================================#
#Use mean value of training data as general predict value
#Setting National day tag by mutiplying correlate coefficient
#=============================================================#

for i in range(15, 29):
	mean = train[train.columns[i]].mean()
	for j in range(len(test)):
		test.ix[j, i] = mean * 0.9
	test.ix[119, i] = test.ix[119, i] * 1.2
	test.ix[120, i] = test.ix[120, i] * 1.5
	test.ix[121, i] = test.ix[121, i] * 1.8
	test.ix[122, i] = test.ix[122, i] * 2.1
	test.ix[123, i] = test.ix[123, i] * 2.4
	test.ix[124, i] = test.ix[124, i] * 2.7
	test.ix[125, i] = test.ix[125, i] * 2.7
	test.ix[126, i] = test.ix[126, i] * 2.4
	test.ix[127, i] = test.ix[127, i] * 2.1
	test.ix[128, i] = test.ix[128, i] * 1.8
	test.ix[129, i] = test.ix[129, i] * 1.5
	test.ix[130, i] = test.ix[130, i] * 1.2

test.to_csv('result.csv', header = None, index = None)