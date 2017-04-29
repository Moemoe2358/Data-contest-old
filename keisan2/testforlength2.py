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
lel, ael, alel, llel = [], [], [], []
# p = 0

for i in range(len(test)):
	pid = test['pid'][i]
	area = test['area'][i]
	location = test['location'][i]
	natural = test['natural_lawson_store'][i]

	temp = train[(train['pid'] == pid) & (train['area'] == area) & (train['location'] == location) & (train['natural_lawson_store'] == natural)]
	temp = temp.reset_index(drop = True)
	length = len(temp)

	# if length == 0:
	# 	submit["b"][i] = 2.0

	# if length == 1:
	# 	submit["b"][i] = temp["y"][length - 1] * 1.5

	# if length == 2:
	# 	submit["b"][i] = temp["y"][length - 1]

	# if length >= 3:
	# 	submit["b"][i] = weight[natural][location][area] * temp["y"][length - 1] + (1 - weight[natural][location][area]) * temp["y"][length - 2]

	# if natural == 1:
	# 	submit["b"][i] = submit["b"][i] * 1.1

	if length == 12 or length < 3:
		continue

	# length -= p

	# le = (np.log(temp['y'][length - 1] + 1) - np.log(temp['y'][length - 2] + 1)) ** 2
	# lel.append(le)
	ale = (np.log(temp['y'][2] + 1) - np.log(temp['y'][1] * weight[natural][location][area] + temp['y'][0] * 1.4 * (1 - weight[natural][location][area]) + 1)) ** 2
	alel.append(ale)
	# ale = (np.log(temp['y'][length - 1] + 1) - np.log(0.8 * temp['y'][length - 2] + 0.2 * temp['y'][length - 3] + 1)) ** 2
	# alel.append(ale)
	# lle = (np.log(temp['y'][length - 1] + 1) - np.log(temp['y'][length - 3] + 1)) ** 2
	# llel.append(lle)
	# ae = (np.log(temp['y'][length - 1] + 1) - np.log(temp['y'][1:length - 1].mean() + 1)) ** 2
	# ael.append(ae)

# print ""
# print "Len:", 12 - p
# print "Last", round(np.sqrt(np.mean(lel)), 3)
print "Moving Average", round(np.sqrt(np.mean(alel)), 4)
# print "2rd Last", round(np.sqrt(np.mean(llel)), 3)
# print "Average", round(np.sqrt(np.mean(ael)), 3)
# print ""

# submit.to_csv("result.csv", header = None, index = None)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
