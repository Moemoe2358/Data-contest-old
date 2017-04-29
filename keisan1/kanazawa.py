# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import csv

train = pd.read_csv("target/target_train.csv")
test = pd.read_csv("result.csv", header = None)

#=========================================================#
#Use mean value of training data as general predict value
#=========================================================#

mean = train[train.columns[7]].mean()

for i in range(len(test)):
	test.ix[i, 7] = mean

#=========================================================#
#Setting event day tag and syukujitsu by mutiplying correlate coefficient
#=========================================================#

test.ix[4, 7] = test.ix[4, 7] * 1.2
test.ix[5, 7] = test.ix[5, 7] * 1.2
test.ix[6, 7] = test.ix[6, 7] * 1.2
test.ix[47, 7] = test.ix[47, 7] * 1.2
test.ix[48, 7] = test.ix[48, 7] * 1.2
test.ix[49, 7] = test.ix[49, 7] * 1.2
test.ix[68, 7] = test.ix[68, 7] * 1.2
test.ix[69, 7] = test.ix[69, 7] * 1.2
test.ix[70, 7] = test.ix[70, 7] * 1.2
test.ix[71, 7] = test.ix[71, 7] * 1.2
test.ix[72, 7] = test.ix[72, 7] * 1.2
test.ix[73, 7] = test.ix[73, 7] * 1.2
test.ix[74, 7] = test.ix[74, 7] * 1.2
test.ix[75, 7] = test.ix[75, 7] * 1.2
test.ix[76, 7] = test.ix[76, 7] * 1.2
test.ix[110, 7] = test.ix[110, 7] * 1.2
test.ix[111, 7] = test.ix[111, 7] * 1.2
test.ix[112, 7] = test.ix[112, 7] * 1.2
test.ix[113, 7] = test.ix[113, 7] * 1.2
test.ix[114, 7] = test.ix[114, 7] * 1.2
test.ix[131, 7] = test.ix[131, 7] * 1.2
test.ix[132, 7] = test.ix[132, 7] * 1.2
test.ix[133, 7] = test.ix[133, 7] * 1.2
test.ix[173, 7] = test.ix[173, 7] * 1.2
test.ix[174, 7] = test.ix[174, 7] * 1.2
test.ix[175, 7] = test.ix[175, 7] * 1.2
test.ix[179, 7] = test.ix[179, 7] * 1.2
test.ix[180, 7] = test.ix[180, 7] * 1.2
test.ix[181, 7] = test.ix[181, 7] * 1.2

test.to_csv('result.csv', header = None, index = None)

# for i in range(7, 8):
# 	mean = train[train.columns[i]].mean()
# 	for j in range(len(test)):
# 		test.ix[j, i] = mean * 0.9
# 	test.ix[119, i] = test.ix[119, i] * 1.2
# 	test.ix[120, i] = test.ix[120, i] * 1.5
# 	test.ix[121, i] = test.ix[121, i] * 1.8
# 	test.ix[122, i] = test.ix[122, i] * 2.1
# 	test.ix[123, i] = test.ix[123, i] * 2.4
# 	test.ix[124, i] = test.ix[124, i] * 2.7
# 	test.ix[125, i] = test.ix[125, i] * 2.7
# 	test.ix[126, i] = test.ix[126, i] * 2.4
# 	test.ix[127, i] = test.ix[127, i] * 2.1
# 	test.ix[128, i] = test.ix[128, i] * 1.8
# 	test.ix[129, i] = test.ix[129, i] * 1.5
# 	test.ix[130, i] = test.ix[130, i] * 1.2

# test.to_csv('tests.csv', header = None, index = None)