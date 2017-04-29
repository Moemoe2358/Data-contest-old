# -*- coding: utf-8 -*-
# 長さの分布：[3, 42, 316, 3, 71, 177, 5, 1, 169, 33, 7, 32, 1504]

import numpy as np
import pandas as pd
import time

start_time = time.time()

train = pd.read_csv("data3/train.tsv", delimiter = "\t")
test = pd.read_csv("data3/test.tsv", delimiter = "\t")

submit = pd.read_csv("sample_submit.csv", header = None)
submit.columns = ["a", "b"]

for i in range(len(test)):
	pid = test['pid'][i]
	area = test['area'][i]
	location = test['location'][i]
	natural = test['natural_lawson_store'][i]

	temp = train[(train['pid'] == pid) & (train['area'] == area) & (train['location'] == location) & (train['natural_lawson_store'] == natural)]
	temp = temp.reset_index(drop = True)
	length = len(temp)

	if length == 0:
		submit["b"][i] = 2.0

	if length == 1:
		submit["b"][i] = temp["y"][length - 1] * 1.6

	if length == 2:
		submit["b"][i] = temp["y"][1]

	if length >= 3 and length <= 11:
		submit["b"][i] = 0.57 * temp["y"][length - 1] + 0.18 * temp["y"][length - 2] + 0.25 * temp["y"][1:length - 1].mean()

	if length == 12:
		submit["b"][i] = 0.57 * temp["y"][length - 1] + 0.18 * temp["y"][length - 2] + 0.16 * temp["y"][0] + 0.09 * temp["y"][1:10].mean()

	if natural == 1:
		submit["b"][i] = submit["b"][i] * 1.15

	submit["b"][i] = submit["b"][i] * 0.95

submit.to_csv("result.csv", header = None, index = None)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"