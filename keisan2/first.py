# -*- coding: utf-8 -*-
# ナチュラルローソン菓子お買得セール　2016年6月7日（火）から2016年6月20日（月）まで
# [3, 42, 316, 3, 71, 177, 5, 1, 169, 33, 7, 32, 1504]

import pandas as pd
import numpy as np
from sklearn import linear_model
import time

start_time = time.time()

train = pd.read_csv("data3/train.tsv", delimiter = '\t')
test = pd.read_csv("data3/test.tsv", delimiter = '\t')

submit = pd.read_csv("sample_submit.csv", header = None)
submit.columns = ["a", "b"]
# count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(test)):
	pid = test["pid"][i]
	area = test["area"][i]
	location = test["location"][i]
	natural = test["natural_lawson_store"][i]

	temp = train[(train.pid == pid) & (train.area == area) & (train.location == location) & (train.natural_lawson_store == natural)]
	temp = temp.reset_index(drop = True)
	
	# count[len(temp)] += 1
	if len(temp) == 0:
		submit["b"][i] = 2.0

	if len(temp) > 0:
		submit["b"][i] = temp["y"][len(temp) - 1]

	if len(temp) == 1:
		submit["b"][i] = submit["b"][i] * 1.6

	if natural == 1:
		submit["b"][i] = submit["b"][i] * 1.05
	else:	
		submit["b"][i] = submit["b"][i] * 0.97

	# XVal = np.empty((0, 2), int)
	# yVal = np.array([])

	# if len(temp) == 12:
	# 	XVal = np.empty((0, 1), float)
	# 	yVal = np.array([])

	# 	for j in range(11):
	# 		XVal = np.append(XVal, np.array([[temp["y"][j]]]), axis = 0)
	# 		yVal = np.append(yVal, np.array([temp["y"][j + 1]]))

	# 		lr = linear_model.LinearRegression()
	# 		lr.fit(XVal, yVal)

	# 	yHatVal = map(lr.predict, XVal)
	# 	bias = 0
	# 	for j in range(len(yHatVal)):
	# 		bias += abs(yHatVal[j] - yVal[j])
	# 	accVal = float(bias)

	# 	if accVal < 1:
	# 		submit["b"][i] = float(map(lr.predict, np.array([temp["y"][10]]))[0])
	# 		print i, pid, area, location, natural, accVal, submit["b"][i]

# print float(countb) / counta
# print count

submit.to_csv("result.csv", header = None, index = None)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
