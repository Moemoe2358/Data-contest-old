# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time

start_time = time.time()

xel, lel, ael, llel, qel = [], [], [], [], []

x0 = np.linspace(1, 7, 7)
x1 = np.linspace(1, 12, 12)

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
		submit["b"][i] = 0.75 * temp["y"][1] + 1.6 * 0.25 * temp["y"][0]

	if length >= 3:
		submit["b"][i] = 0.75 * temp["y"][length - 1] + 0.25 * temp["y"][length - 2]

	if natural == 1:
		submit["b"][i] = submit["b"][i] * 1.1

	if length < 12:
		continue

	r = np.array(temp['y'][:7])
	z = np.polyfit(x0, r, 4)
	p = np.poly1d(z)

	l = []
	pel = []
	for j in range(12):
		if p(j + 1) <= 0:
			l.append(1)
		else:
			l.append(p(j + 1))

	for j in range(7, 11):
		pel.append((np.log(temp['y'][j] + 1) - np.log(l[j] + 1)) ** 2)

	if np.sqrt(np.mean(pel)) < 0.1:
		print i, round(np.sqrt(np.mean(pel)), 4), round(abs(np.log(temp['y'][11] + 1) - np.log(l[11] + 1)), 4)

		r = np.array(temp['y'])
		z = np.polyfit(x1, r, 4)
		p = np.poly1d(z)

		ans = p(13)
		if ans > 0.5:
			submit["b"][i] = ans

	le = (np.log(temp['y'][11] + 1) - np.log(temp['y'][10] + 1)) ** 2
	lel.append(le)
	xe = (np.log(temp['y'][11] + 1) - np.log(0.5 * temp['y'][10] + 0.5 * temp['y'][9] + 1)) ** 2
	xel.append(xe)
	lle = (np.log(temp['y'][11] + 1) - np.log(0.75 * temp['y'][10] + 0.25 * temp['y'][9] + 1)) ** 2
	llel.append(lle)
	ae = (np.log(temp['y'][11] + 1) - np.log(temp['y'][1:11].mean() + 1)) ** 2
	ael.append(ae)
	if np.sqrt(np.mean(pel)) < 0.1:
		qe = (np.log(temp['y'][11] + 1) - np.log(l[11] + 1)) ** 2
		qel.append(qe)

print ""
print "Last", round(np.sqrt(np.mean(lel)), 4)
print "Fixed weight(0.5, 0.5)", round(np.sqrt(np.mean(xel)), 4)
print "Fixed weight(0.75, 0.25)", round(np.sqrt(np.mean(llel)), 4)
print "Average", round(np.sqrt(np.mean(ael)), 4)
print "Regression", round(np.sqrt(np.mean(qel)), 4), len(qel)
print ""

submit.to_csv("result.csv", header = None, index = None)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"