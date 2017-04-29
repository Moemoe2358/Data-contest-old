# -*- coding: utf-8 -*-
# 長さの分布：[3, 42, 316, 3, 71, 177, 5, 1, 169, 33, 7, 32, 1504]

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import time

start_time = time.time()

train = pd.read_csv("data3/train.tsv", delimiter = "\t")
test = pd.read_csv("data3/test.tsv", delimiter = "\t")

submit = pd.read_csv("sample_submit.csv", header = None)
submit.columns = ["a", "b"]

# locationset = ["学校立地", "観光立地", "ビジネス立地", "住宅立地"]
# naturalset = [0, 1]
# areaset = ["四国", "北海道", "関東", "四国", "近畿", "東北", "中部", "九州"]

# lag = 30
# end = 100 - lag + 1

for g in range(10):
	for h in range(10 - g):
		floatg = g * 0.1
		floath = h * 0.1
		l = []
		for i in range(len(test)):
			pid = test['pid'][i]
			area = test['area'][i]
			location = test['location'][i]
			natural = test['natural_lawson_store'][i]

			temp = train[(train['pid'] == pid) & (train['area'] == area) & (train['location'] == location) & (train['natural_lawson_store'] == natural)]
			temp = temp.reset_index(drop = True)

			if len(temp) < 12:
				continue

			for j in range(3, 11):
				l.append((np.log(temp['y'][j] + 1) - np.log(floatg * temp['y'][j - 1] + floath * temp['y'][j - 2] + (1 - floatg - floath) * temp['y'][j + 1] + 1)) ** 2)

		print floatg, floath, 1 - floatg - floath, round(np.sqrt(np.mean(l)), 4)
				

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"