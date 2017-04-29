# -*- coding: utf-8 -*-
# 長さの分布：[3, 42, 316, 3, 71, 177, 5, 1, 169, 33, 7, 32, 1504]

import numpy as np
import pandas as pd
from sklearn import linear_model 
import time

start_time = time.time()

train = pd.read_csv("data3/train.tsv", delimiter = "\t")
test = pd.read_csv("data3/test.tsv", delimiter = "\t")

submit = pd.read_csv("sample_submit.csv", header = None)
submit.columns = ["a", "b"]

weight = {0:{"観光立地":{"関東":0.97, "近畿":0.96, "東北":0.92, "四国":1.01, "北海道":0.68, "九州":0.96, "中部":1.08, "中国":1.04}, \
"住宅立地":{"関東":0.97, "近畿":0.96, "東北":0.76, "四国":0.56, "北海道":0.95, "九州":0.94, "中部":1.02, "中国":0.78}, \
"ビジネス立地":{"関東":0.96, "近畿":0.97, "東北":0.88, "四国":0.75, "北海道":0.8, "九州":0.99, "中部":0.94, "中国":0.88}, \
"学校立地":{"関東":1, "近畿":0.88, "東北":0.63, "四国":0.59, "北海道":0.48, "九州":0.72, "中部":0.82, "中国":0.62}}, \
1:{"観光立地":{"関東":0.87}, "住宅立地":{"関東":0.9}, "ビジネス立地":{"関東":0.88}}}

# weight = {0:{"観光立地":{"関東":0.8, "近畿":0.8, "東北":0.69, "四国":0.66, "北海道":0.69, "九州":0.65, "中部":0.72, "中国":0.79}, \
# "住宅立地":{"関東":0.72, "近畿":0.78, "東北":0.74, "四国":0.52, "北海道":0.76, "九州":0.67, "中部":0.74, "中国":0.73}, \
# "ビジネス立地":{"関東":0.72, "近畿":0.97, "東北":0.88, "四国":0.75, "北海道":0.76, "九州":0.99, "中部":0.94, "中国":0.68}, \
# "学校立地":{"関東":1, "近畿":0.88, "東北":0.63, "四国":0.59, "北海道":0.48, "九州":0.72, "中部":0.82, "中国":0.62}}, \
# 1:{"観光立地":{"関東":0.69}, "住宅立地":{"関東":0.92}, "ビジネス立地":{"関東":0.88}}}

item = [0.36, 0.44, 0.57, 0.59, 0.56, 0.76, 0.9, 0.51, 0.8, 0.8, \
0.86, 0.73, 0.67, 0.8, 0.61, 0.75, 0.44, 0.77, 0.54, 0.75, \
0.62, 0.56, 0.51, 0.95, 0.75, 0.8, 0.79, 0.59, 0.6, 0.75, \
0.48, 0.8, 0.58, 0.89, 0.68, 0.7, 0.99, 0.8, 0.57, 0.52, \
0.8, 0.75, 0.48, 0.65, 0.75, 0.8, 0.72, 0.71, 0.93, 0.81, \
0.91, 0.85, 0.91, 0.73, 0.8, 0.61, 0.53, 0.75, 0.88, 0.73, \
0.6, 0.8, 0.62, 0.67, 0.63, 0.77, 0.71, 0.67, 0.55, 0.66, \
0.85, 0.8, 0.96, \
]

xel, lel, ael, llel = [], [], [], []
# p = 0

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
		submit["b"][i] = 0.75 * temp["y"][1] + 1.6 * 0.25 * temp["y"][0]

	if length >= 3:
		submit["b"][i] = 0.75 * temp["y"][length - 1] + 0.25 * temp["y"][length - 2]

	if natural == 1:
		submit["b"][i] = submit["b"][i] * 1.1

	if length < 4:
		continue

	le = (np.log(temp['y'][length - 1] + 1) - np.log(temp['y'][length - 2] + 1)) ** 2
	lel.append(le)
	xe = (np.log(temp['y'][length - 1] + 1) - np.log(0.5 * temp['y'][length - 2] + 0.5 * temp['y'][length - 3] + 1)) ** 2
	xel.append(xe)
	lle = (np.log(temp['y'][length - 1] + 1) - np.log(0.75 * temp['y'][length - 2] + 0.25 * temp['y'][length - 3] + 1)) ** 2
	llel.append(lle)
	ae = (np.log(temp['y'][length - 1] + 1) - np.log(temp['y'][1:length - 1].mean() + 1)) ** 2
	ael.append(ae)

print ""
print "Last", round(np.sqrt(np.mean(lel)), 4)
print "Fixed weight(0.5, 0.5)", round(np.sqrt(np.mean(xel)), 4)
print "Fixed weight(0.75, 0.25)", round(np.sqrt(np.mean(llel)), 4)
print "Average", round(np.sqrt(np.mean(ael)), 4)
print ""

submit.to_csv("result.csv", header = None, index = None)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
