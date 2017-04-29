# -*- coding: utf-8 -*-
# ナチュラルローソン菓子お買得セール　2016年6月7日（火）から2016年6月20日（月）まで
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

name1 = "sns/菓子種別/"
name2 = ["クッキー", "クラッカー", "スナック", "せんべい", "チョコレート", "ドーナツ", "パイ", "ビスケット", "ポテトチップス", "マシュマロ", \
"ラスク", "焼菓子", "中華菓子", "半生菓子", "米菓子", "油菓子", "洋菓子", "和菓子"]
name3 = "_twitter"
name4 = ".tsv"

pointer = [2, 2, 2, 2, 6, 8, 4, 2, 4, 7, \
0, 1, 1, 2, 7, 7, 2, 2, 7, 10, \
9, 0, 2, 2, 2, 0, 5, 2, 2, 11, \
2, 15, 2, 2, 2, 15, 5, 13, 2, 11, \
4, 2, 2, 2, 2, 2, 4, 0, 4, 4, \
4, 2, 2, 2, 14, 2, 2, 2, 4, 2, \
2, 11, 14, 8, 8, 14, 14, 2, 2, 6, \
15, 11, 4]

calendar = [30, 31, 31, 30, 31, 30, 31, 31, 29, 31, 30, 31]
vel, lel, ael = [], [], []

for i in range(len(test)):
	pid = test['pid'][i]
	area = test['area'][i]
	location = test['location'][i]
	natural = test['natural_lawson_store'][i]

	temp = train[(train['pid'] == pid) & (train['area'] == area) & (train['location'] == location) & (train['natural_lawson_store'] == natural)]
	temp = temp.reset_index(drop = True)
	length = len(temp)

	# if length > 0:
	# 	submit["b"][i] = temp["y"][length - 1]

	# if length == 1:
	# 	submit["b"][i] = submit["b"][i] * 1.5

	# if natural == 1:
	# 	submit["b"][i] = submit["b"][i] * 1.05

	if length == 0:
		submit["b"][i] = 2.0

	if length == 1:
		submit["b"][i] = temp["y"][length - 1] * 1.6

	if length == 2:
		submit["b"][i] = 0.75 * temp["y"][1] + 1.6 * 0.25 * temp["y"][0]

	if length >= 3:
		submit["b"][i] = 0.75 * temp["y"][length - 1] + 0.25 * temp["y"][length - 2]

	# if length >= 4:
	# 	submit["b"][i] = 0.70 * temp["y"][length - 1] + 0.20 * temp["y"][length - 2] + 0.10 * temp["y"][length - 3]
		# submit["b"][i] = (temp['y'].max() + temp['y'].min() + temp['y'].median() * 4) / 6.0

	if natural == 1:
		submit["b"][i] = submit["b"][i] * 1.1

	if length < 12:
		continue

	s = int(pid[1:])
	filename = name1 + name2[pointer[s]] + name3 + name4
	sns_data = pd.read_csv(filename, delimiter = "\t")
	temp_data = sns_data

	l = []
	for j in range(length):
		tempm = temp_data.head(calendar[j])["freq"].mean()
		l.append(tempm)
		temp_data = temp_data[temp_data.index >= calendar[j]]
		temp_data = temp_data.reset_index(drop = True)

	X = np.matrix(l)
	print X
	XTrain = X[:, :11]
	XTrain = XTrain.reshape(11, 1)
	XVal = X[:, 11]
	XTest = X[:, 11]

	yTrain = np.array(temp['y'][1:12])
	print yTrain
	# lr = linear_model.RidgeCV()
	lr = linear_model.LinearRegression()
	lr.fit(XTrain, yTrain)

	yHatTrain = map(lr.predict, XTrain)
	yHatVal = map(lr.predict, XVal)
	yHatTest = map(lr.predict, XTest)

	te = np.mean(np.abs(np.squeeze(np.asarray(yHatTrain)) - yTrain))
	# print i, te
	if te < 0.2:
		print i, "ww"
		submit["b"][i] = yHatTest[0]

	# ve = ((np.log(temp['y'][11] + 1) - np.log(yHatVal[0] + 1)) ** 2)[0]
	# vel.append(ve)
	# le = (np.log(temp['y'][11] + 1) - np.log(temp['y'][10] + 1)) ** 2
	# lel.append(le)
	# ae = (np.log(temp['y'][11] + 1) - np.log(temp['y'][:11].mean() + 1)) ** 2
	# ael.append(ae)

	# print i, ve, le, ae



# print "Linear", np.sqrt(np.mean(vel)), len(vel)
# print "Last", np.sqrt(np.mean(lel)), len(lel)
# print "Average", np.sqrt(np.mean(ael)), len(ael)

submit.to_csv("result2.csv", header = None, index = None)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"


