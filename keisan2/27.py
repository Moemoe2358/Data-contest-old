# -*- coding: utf-8 -*-
# 長さの分布：[3, 42, 316, 3, 71, 177, 5, 1, 169, 33, 7, 32, 1504]

import numpy as np
import pandas as pd
import datetime
import csv
from sklearn import linear_model 
import time

start_time = time.time()

train = pd.read_csv("data3/train.tsv", delimiter = "\t")
test = pd.read_csv("data3/test.tsv", delimiter = "\t")

submit = pd.read_csv("sample_submit.csv", header = None)
submit.columns = ["a", "b"]

lel, ael, alel, llel, fel = [], [], [], [], []
p = 0

for i in range(len(test)):
	pid = test['pid'][i]
	area = test['area'][i]
	location = test['location'][i]
	natural = test['natural_lawson_store'][i]

	temp = train[(train['pid'] == pid) & (train['area'] == area) & (train['location'] == location) & (train['natural_lawson_store'] == natural) ]
	temp = temp.reset_index(drop = True)
	length = len(temp)

	if length == 0:
		submit["b"][i] = 2.0

	if length == 1:
		submit["b"][i] = temp["y"][length - 1] * 1.5

	if length == 2:
		submit["b"][i] = temp["y"][length - 1]

	if length >= 3:
		submit["b"][i] = 0.67 * temp["y"][length - 1] + 0.33 * temp["y"][length - 2]

	if natural == 1:
		submit["b"][i] = submit["b"][i] * 1.07

	if length < 12:
		continue

	length -= p
	
	# best = 100
	# for j in range(11):
	# 	fe = []
	# 	for k in range(3, length - 1):
	# 		fe.append((np.log(temp['y'][k - 1] + 1) - np.log(0.1 * j * temp['y'][k - 2] + (1 - j * 0.1) * temp['y'][k - 3] + 1)) ** 2)
	# 	fem = np.mean(fe)
	# 	print fem
	# 	if fem < best:
	# 		best = fem
	# 		note = j
	# print i, 0.1 * note, 1 - 0.1 * note
	# fel.append((np.log(temp['y'][length - 1] + 1) - np.log(0.1 * note * temp['y'][length - 2] + (1 - note * 0.1) * temp['y'][length - 3] + 1)) ** 2)

	le = (np.log(temp['y'][length - 1] + 1) - np.log(temp['y'][length - 2] + 1)) ** 2
	lel.append(le)
	ale = (np.log(temp['y'][length - 1] + 1) - np.log(0.7 * temp['y'][length - 2] + 0.3 * temp['y'][length - 3] + 0 * temp['y'][length - 4] + 1)) ** 2
	alel.append(ale)
	lle = (np.log(temp['y'][length - 1] + 1) - np.log(temp['y'][length - 3] + 1)) ** 2
	llel.append(lle)
	ae = (np.log(temp['y'][length - 1] + 1) - np.log(temp['y'][1:length - 1].mean() + 1)) ** 2
	ael.append(ae)

print ""
print "Len:", 12 - p
print "Last", round(np.sqrt(np.mean(lel)), 3)
print "Moving Average", round(np.sqrt(np.mean(alel)), 3)
print "2rd Last", round(np.sqrt(np.mean(llel)), 3)
print "Average", round(np.sqrt(np.mean(ael)), 3)
# print "Final", round(np.sqrt(np.mean(fel)), 3)
print ""

submit.to_csv("result.csv", header = None, index = None)

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
