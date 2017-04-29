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

lll = []
locationset = ["住宅立地", "観光立地", "ビジネス立地", "学校立地"]
naturalset = [0, 1]
areaset = ["中国", "北海道", "関東", "四国", "近畿", "東北", "中部", "九州"]

lag = 50
end = 100 - lag + 1

for x1 in locationset:
	for x2 in naturalset:
		# for x3 in areaset:
		for h in range(end):

			floath = (h + lag) * 0.01
			ll = []

			for i in range(len(test)):
				pid = test['pid'][i]
				area = test['area'][i]
				location = test['location'][i]
				natural = test['natural_lawson_store'][i]

				temp = train[(train['pid'] == pid) & (train['area'] == area) & (train['location'] == location) & (train['natural_lawson_store'] == natural)]
				temp = temp.reset_index(drop = True)

				length = len(temp)

				if length < 4 or location != x1 or natural != x2:
					continue

				l = []
				for j in range(3, length):
					l.append((np.log(temp['y'][j - 1] + 1) - np.log(floath * temp['y'][j - 2] + (1 - floath) * temp['y'][j - 3] + 1)) ** 2)

				ll.append(np.mean(l))
			
			if len(ll) == 0:
				print x1, x2, "NaN"
				break
			lll.append(np.sqrt(np.mean(ll)))
			print h + lag, lll[h]
			if h > 2:
				if lll[h - 1] < lll[h] and lll[h - 1] < lll[h - 2]:
					print x1, x2, h - 1 + lag, lll[h]
					lll = []
					break
		lll = []

# x = range(86, 101)
# plt.plot(x, lll)
# plt.show()

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"


# Len: 12
# Last 0.16
# Moving Average 0.167 0.163 0.16
# Moving Average* 0.162 0.157
# 2rd Last 0.203
# Average 0.149
# Final 0.163

# Len: 11
# Last 0.156
# Moving Average 0.139 0.14 0.142
# Moving Average* 0.141 0.139
# 2rd Last 0.159
# Average 0.144
# Final 0.144

# Len: 10
# Last 0.149
# Moving Average 0.139 0.138 0.139 
# Moving Average* 0.139 0.137
# 2rd Last 0.16
# Average 0.158
# Final 0.142

# Len: 9
# Last 0.136
# Moving Average 0.124 0.124 0.125
# Moving Average* 0.119 0.119
# 2rd Last 0.147
# Average 0.129

# Len: 8
# Last 0.137
# Moving Average 0.125 0.125 0.126
# Moving Average* 0.132 0.127
# 2rd Last 0.149
# Average 0.161

# Len: 7
# Last 0.133
# Moving Average 0.141 0.136 0.133
# Moving Average* 0.152 0.142
# 2rd Last 0.186
# Average 0.164

# Len: 6
# Last 0.153
# Moving Average 0.151 0.148 0.146
# Moving Average* 0.155 0.145
# 2rd Last 0.197
# Average 0.149

# Len: 5
# Last 0.176
# Moving Average 0.179 0.175 0.173
# Moving Average* 0.166 0.163
# 2rd Last 0.215
# Average 0.166
