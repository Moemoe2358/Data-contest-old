# -*- coding: utf-8 -*-
# 長さの分布：[3, 42, 316, 3, 71, 177, 5, 1, 169, 33, 7, 32, 1504]

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time

start_time = time.time()

train = pd.read_csv("data3/train.tsv", delimiter = "\t")
test = pd.read_csv("data3/test.tsv", delimiter = "\t")

submit = pd.read_csv("sample_submit.csv", header = None)
submit.columns = ["a", "b"]

lll = []

for h in range(70, 90):
	floath = h * 0.01
	ll = []
	for i in range(len(test)):
		pid = test['pid'][i]
		area = test['area'][i]
		location = test['location'][i]
		natural = test['natural_lawson_store'][i]

		temp = train[(train['pid'] == pid) & (train['area'] == area) & (train['location'] == location) & (train['natural_lawson_store'] == natural)]
		temp = temp.reset_index(drop = True)

		if len(temp) < 4:
			continue

		l = []

		for j in range(2, len(temp) - 1):
			l.append((np.log(temp['y'][j] + 1) - np.log(floath * temp['y'][j - 1] + (1 - floath) * temp['y'][j - 2] + 1)) ** 2)

		ll.append(np.mean(l))

	print floath, 1 - floath, round(np.sqrt(np.mean(ll)), 5)
	lll.append(np.sqrt(np.mean(ll)))

x = range(70, 90)
plt.plot(x, lll)
plt.show()

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"