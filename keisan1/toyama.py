# -*- coding: utf-8 -*-

#keyword

#--------------------------------------------------------------------------#

#--------------------------------------------------------------------------#
#package
import numpy as np
import pandas as pd
import datetime
import csv
from sklearn import linear_model
#--------------------------------------------------------------------------#
#build model for young person, mid-aged person, old person respectively
#--------------------------------------------------------------------------#
#自動化
pd.options.mode.chained_assignment = None
yid = ['16201_若年層','16201_中年層','16201_老年層']
colid = [62, 63, 64]
weaid = ['w000017', 'w000017', 'w000017']
geoid = ['16201', '16201', '16201']
snsid = [
['date', '観光_blog'],
['date', '観光_blog'],
['date', '観光_blog'],
]
#--------------------------------------------------------------------------#
# "ho" means the beginning and the end of the holidays
#--------------------------------------------------------------------------#

ho = ["2014-07-19", "2014-07-21", "2014-08-09", "2014-08-15"
, "2014-09-13",  "2014-09-15", "2014-09-20", "2014-09-23", "2014-10-11"
, "2014-10-13", "2014-11-01", "2014-11-03", "2014-11-22", "2014-11-24", "2014-12-27", "2014-12-28", "2014-12-29"
, "2014-12-30", "2015-01-03", "2015-01-04", "2015-01-10", "2015-01-12"
, "2015-05-02", "2015-07-18", "2015-07-20"
, "2015-08-08", "2015-08-16"
, "2015-09-19", "2015-09-23", "2015-10-10", "2015-10-12"
, "2015-11-21", "2015-11-23"]

#--------------------------------------------------------------------------#
# "hoho" means the middle of holidays
#--------------------------------------------------------------------------#

hoho = ["2014-07-20", "2014-08-10", "2014-08-11", "2014-08-12", "2014-08-13", "2014-08-14"
 "2014-09-14", "2014-09-21", "2014-09-22", "2014-10-12", "2014-11-02", "2014-11-23"
, "2014-12-31", "2015-01-01", "2015-01-02", "2015-01-11", "2015-05-03", "2015-05-04", "2015-05-05"
, "2015-07-19"
, "2015-08-09", "2015-08-10", "2015-08-11", "2015-08-12", "2015-08-13", "2015-08-14", "2015-08-15"
, "2015-09-20", "2015-09-21", "2015-09-22", "2015-10-11", "2015-11-22"]

# hoho = []
#--------------------------------------------------------------------------#
# event days in toyama
#--------------------------------------------------------------------------#
toyama = ["2014-09-01", "2014-09-02", "2014-09-03", "2015-02-24", "2015-02-25", "2015-03-14", "2015-03-21", "2015-03-28", "2015-05-30"
, "2015-09-01", "2015-09-02", "2015-09-03", "2015-11-14", "2015-11-15"]

# kanazawa = ["2014-06-06", "2014-06-07", "2014-06-08"
# , "2014-11-22", "2014-11-23", "2014-11-24", "2014-11-29", "2014-11-30"
# , "2015-06-05", "2015-06-06", "2015-06-07"
# , "2015-09-19", "2015-09-20", "2015-09-21", "2015-09-22"
# , "2015-11-20", "2015-11-21", "2015-11-22", "2015-11-27", "2015-11-28", "2015-11-29"
# ]

# kanazawa = ["2014-07-19", "2014-07-21", "2014-08-09", "2014-08-15"
# , "2014-09-13",  "2014-09-15", "2014-09-20", "2014-09-23", "2014-10-11"
# , "2014-10-13", "2014-11-01", "2014-11-03", "2014-11-22", "2014-11-24", "2014-12-27", "2014-12-28", "2014-12-29"
# , "2014-12-30", "2015-01-03", "2015-01-04", "2015-01-10", "2015-01-12"
# , "2015-05-02", "2015-07-18", "2015-07-20"
# , "2015-08-08", "2015-08-16"
# , "2015-09-19", "2015-09-23", "2015-10-10", "2015-10-12"
# , "2015-11-21", "2015-11-23"
# , "2014-07-20", "2014-08-10", "2014-08-11", "2014-08-12", "2014-08-13", "2014-08-14"
# , "2014-09-14", "2014-09-21", "2014-09-22", "2014-10-12", "2014-11-02", "2014-11-23"
# , "2014-12-31", "2015-01-01", "2015-01-02", "2015-01-11", "2015-05-03", "2015-05-04", "2015-05-05"
# , "2015-07-19"
# , "2015-08-09", "2015-08-10", "2015-08-11", "2015-08-12", "2015-08-13", "2015-08-14", "2015-08-15"
# , "2015-09-20", "2015-09-21", "2015-09-22", "2015-10-11", "2015-11-22"
# , "2014-06-06", "2014-06-07", "2014-06-08"
# , "2014-11-22", "2014-11-23", "2014-11-24", "2014-11-29", "2014-11-30"
# , "2015-06-05", "2015-06-06", "2015-06-07"
# , "2015-09-19", "2015-09-20", "2015-09-21", "2015-09-22"
# , "2015-11-20", "2015-11-21", "2015-11-22", "2015-11-27", "2015-11-28", "2015-11-29"
# ]
#金沢城・兼六園ライトアップ〜初夏の段〜（6月5日〜6月7日19：00〜21：00開催）
#金沢城・兼六園ライトアップ〜秋の段〜（11月20日〜11月29日開催）

# kanazawa = []
for h in range(len(yid)):

#--------------------------------------------------------------------------#
#train_prepare
#--------------------------------------------------------------------------#

	train_all = pd.read_csv("target/target_train.csv")
	train_data = pd.DataFrame(train_all['date'])

#--------------------------------------------------------------------------#
# Add train weekdayflag, shikannsennflag, holidayflag, mid-holidayflag, eventflag
# and turn them into dummy variable
#--------------------------------------------------------------------------#
	train_data = train_data.assign(weekday = 'NA')
	train_data = train_data.assign(hoho = 'NA')
	train_data = train_data.assign(holiday = 'NA')
	train_data = train_data.assign(shikannsenn = 'NA')
	train_data = train_data.assign(event = 'NA')
	for i in range(len(train_data)):
		a = datetime.datetime.strptime(train_data['date'][i], "%Y-%m-%d")
		a = datetime.date(a.year, a.month, a.day)
		if a >= datetime.date(2015, 3, 14):
			train_data['shikannsenn'][i] = 1
		else:
			train_data['shikannsenn'][i] = 0
		if str(a) in ho:
			train_data['holiday'][i] = 1
		else:
			train_data['holiday'][i] = 0
		if str(a) in hoho:
			train_data['hoho'][i] = 1
		else:
			train_data['hoho'][i] = 0
		if h < 3:
			if str(a) in toyama:
				train_data['event'][i] = 1
			else:
				train_data['event'][i] = 0
		else:
			if str(a) in kanazawa:
				train_data['event'][i] = 1
			else:
				train_data['event'][i] = 0			
		train_data['weekday'][i] = a.isoweekday()
		# if a.isoweekday() == 6 or a.isoweekday() == 7:
		# 	train_data['isweekday'][i] = 1
		# else:
		# 	train_data['isweekday'][i] = 0

	dum = pd.get_dummies(train_data["weekday"])
	train_data = pd.concat((train_data, dum), axis=1)
	train_data = train_data.drop(['weekday'], axis=1)
	# if a.isoweekday() == 6 or a.isoweekday() == 7:
	# 	train_data['isweekday'][i] = 2
	# else:
	# 	if a.isoweekday() == 1 or a.isoweekday() == 5:
	# 		train_data['isweekday'][i] = 1
	# 	else:
	# 		train_data['isweekday'][i] = 0
#--------------------------------------------------------------------------#
#train_weather
#--------------------------------------------------------------------------#
	weather = pd.read_csv("weather/weather_train.csv")
	weather = weather[weather.ID == weaid[h]]
	b = weather[['date','shine']]
	train_data = pd.merge(train_data, b, left_on='date', right_on='date', how='outer')
#--------------------------------------------------------------------------#
#train_geosns
#--------------------------------------------------------------------------#

	# geo = pd.read_csv("geo_location/geo_location_train.csv")
	# c = geo[['date',geoid[h]]]
	# train_data = pd.merge(train_data, c, left_on='date', right_on='date', how='outer')
#--------------------------------------------------------------------------#
#train_snskeyword
#--------------------------------------------------------------------------#
	sns = pd.read_csv("sns/sns_train.csv")
	ll = list(sns.columns.values)
	l = []
	for i in range(1, len(ll)):
		cor = np.corrcoef(train_all[yid[h]], np.log(sns[ll[i]]))[0, 1]
		if cor > -1:
			l.append([cor, ll[i]])
	l.sort()
	for i in range(len(l)):
		print l[i][1], l[i][0]
	d = sns[snsid[h]]

	for i in range(1, len(d)):
		d.iloc[:,i:i + 1] = np.log(d.iloc[:,i:i + 1])

	train_data = pd.merge(train_data, d, left_on='date', right_on='date', how='outer')
#--------------------------------------------------------------------------#
#train_sensor
#--------------------------------------------------------------------------#
# sen = pd.read_csv("sensor/sensor_train.csv")
# sen = sen[sen.ID == 33149937]
# e = sen[['date','wet0','wet6','wet12','wet18']]
# train_data = pd.merge(train_data, e, left_on='date', right_on='date', how='outer')

#--------------------------------------------------------------------------#
# pre-processing, delete outlier data, step 1:delete data outside thres
#--------------------------------------------------------------------------#
	X = np.matrix(train_data.drop(['date'], axis=1))
	y = np.matrix(train_all[yid[h]])
	XVal = X
	yVal = y

	Q1 = np.percentile(y, 75)
	Q3 = np.percentile(y, 25)
	thres = Q1 + 1.2 * (Q1 - Q3)
	
	k = 0
	pre = pd.concat((train_data, train_all[yid[h]]), axis=1)
	while k < len(pre):
		if pre[yid[h]][k] >= thres:
			print pre['date'][k], pre[yid[h]][k]
			pre = pre.drop(pre.index[k])
			pre = pre.reset_index(drop=True)
			k = k - 1
		k = k + 1

	k = 0
	while k < len(X):
		if y[0, k] >= thres:
			X = np.delete(X, k, axis=0)
			y = np.delete(y, k)
			k = k - 1
		k = k + 1

#--------------------------------------------------------------------------#
# pre-processing, delete outlier data, step 2:delete data worse the original model
#--------------------------------------------------------------------------#
	lr = linear_model.RidgeCV()
	thres = 5
	k = 100
	XTrain = X[:k,:]
	yTrain = y[:,:k]

	lr.fit(XTrain,np.squeeze(np.asarray(yTrain)))
	yHatTrain = map(lr.predict, XTrain)
	yHatVal = map(lr.predict, XVal)
	temp = np.mean(np.abs(np.squeeze(np.asarray(yTrain)) - np.squeeze(np.asarray(yHatTrain))))
	
	while k < len(X):
		XTrain = X[:k,:]
		yTrain = y[:,:k]

		lr.fit(XTrain,np.squeeze(np.asarray(yTrain)))
		yHatTrain = map(lr.predict, XTrain)
		yHatVal = map(lr.predict, XVal)
		resultTrain = np.mean(np.abs(np.squeeze(np.asarray(yTrain)) - np.squeeze(np.asarray(yHatTrain))))
		resultVal = np.mean(np.abs(np.squeeze(np.asarray(yVal)) - np.squeeze(np.asarray(yHatVal))))
		if resultTrain - temp > resultTrain / 50:
			k = k - 1
			print resultTrain, resultVal ,resultTrain - temp, k
			X = np.delete(X, (k - 1), axis=0)
			y = np.delete(y, (k - 1))
		else:
			temp = resultTrain
		k = k + 1

	XTrain = X
	yTrain = y
	# XVal = X[N:, :]
	# yVal = y[:, N:]
#--------------------------------------------------------------------------#
# Training fit
#--------------------------------------------------------------------------#
#train_evaluation
	#lr = linear_model.Ridge(alpha = 1)
	#lr = linear_model.LinearRegression()
	#lr = linear_model.ElasticNetCV()
	#lr = linear_model.ElasticNet()
	#lr = linear_model.LassoCV()
	#lr = linear_model.Lasso(alpha = 3)
	lr = linear_model.RidgeCV()

	lr.fit(XTrain,np.squeeze(np.asarray(yTrain)))
	print lr.coef_
	yHatTrain = map(lr.predict, XTrain)
	yHatVal = map(lr.predict, XVal)
	
	# w = reg.ridgeRegres(XTrain, yTrain, 0.2)
	# yHatTrain = np.dot(XTrain,w)
	# yHatVal = np.dot(XVal,w)

	# print colid[h], "Training error ", np.mean(np.abs(yTrain - yHatTrain.T))
	# print colid[h], "Validation error ", np.mean(np.abs(yVal - yHatVal.T))
	te = np.mean(np.abs(np.squeeze(np.asarray(yTrain)) - np.squeeze(np.asarray(yHatTrain))))
	ve = np.mean(np.abs(np.squeeze(np.asarray(yVal)) - np.squeeze(np.asarray(yHatVal))))
	print colid[h], "Training error ", te
	print colid[h], "Validation error ", ve, te / ve
#--------------------------------------------------------------------------#
#test_prepare
#--------------------------------------------------------------------------#

	test_data = pd.read_csv("result.csv", header = None)
	test_data = test_data.iloc[:,0:1]
	test_data.columns = ["date"]
#--------------------------------------------------------------------------#
# Add test weekdayflag, shikannsennflag, holidayflag, mid-holidayflag, eventflag
# and turn them into dummy variable
#--------------------------------------------------------------------------#

	test_data = test_data.assign(weekday = 'NA')
	test_data = test_data.assign(hoho = 'NA')
	test_data = test_data.assign(holiday = 'NA')
	test_data = test_data.assign(shikannsenn = 1)
	test_data = test_data.assign(event = 'NA')
	for i in range(len(test_data)):
		a = datetime.datetime.strptime(test_data['date'][i], "%Y-%m-%d")
		a = datetime.date(a.year, a.month, a.day)
		if str(a) in ho:
			test_data['holiday'][i] = 1
		else:
			test_data['holiday'][i] = 0
		if str(a) in hoho:
			test_data['hoho'][i] = 1
		else:
			test_data['hoho'][i] = 0
		if h < 3:
			if str(a) in toyama:
				test_data['event'][i] = 1
			else:
				test_data['event'][i] = 0
		else:
			if str(a) in kanazawa:
				test_data['event'][i] = 1
			else:
				test_data['event'][i] = 0
		test_data['weekday'][i] = a.isoweekday()
		# if a.isoweekday() == 6 or a.isoweekday() == 7:
		# 	test_data['isweekday'][i] = 1
		# else:
		# 	test_data['isweekday'][i] = 0
	
	dum = pd.get_dummies(test_data["weekday"])
	test_data = pd.concat((test_data, dum), axis=1)
	test_data = test_data.drop(['weekday'], axis=1)

	# if a.isoweekday() == 6 or a.isoweekday() == 7:
	# 	test_data['isweekday'][i] = 2
	# else:
	# 	if a.isoweekday() == 1 or a.isoweekday() == 5:
	# 		test_data['isweekday'][i] = 1
	# 	else:
	# 		test_data['isweekday'][i] = 0
#--------------------------------------------------------------------------#
#test_weather
#--------------------------------------------------------------------------#

	weathert = pd.read_csv("weather/weather_test.csv")
	weathert = weathert[weathert.ID == weaid[h]]
	b = weathert[['date','shine']]
	test_data = pd.merge(test_data, b, left_on='date', right_on='date', how='outer')
#--------------------------------------------------------------------------#

#--------------------------------------------------------------------------#
#test_geosns
	# geot = pd.read_csv("geo_location/geo_location_test.csv")
	# c = geot[['date',geoid[h]]]
	# test_data = pd.merge(test_data, c, left_on='date', right_on='date', how='outer')
#--------------------------------------------------------------------------#
#test_snskeyword
#--------------------------------------------------------------------------#

	snst = pd.read_csv("sns/sns_test.csv")
	d = snst[snsid[h]]
	numFeat = np.shape(d)[1]

#--------------------------------------------------------------------------#
#replace_nan value
#--------------------------------------------------------------------------#

	for i in range(1, numFeat):
		meanVal = np.median(d.iloc[:, i:i + 1])
		d.iloc[:,i:i + 1] = d.fillna(meanVal)

	for i in range(1, len(d)):
		d.iloc[:,i:i + 1] = np.log(d.iloc[:,i:i + 1])

	test_data = pd.merge(test_data, d, left_on='date', right_on='date', how='outer')
#--------------------------------------------------------------------------#

#--------------------------------------------------------------------------#
#test_sensor
# sent = pd.read_csv("sensor/sensor_test.csv")
# sent = sent[sent.ID == 33149937]
# e = sent[['date','wet0','wet6','wet12','wet18']]
# test_data = pd.merge(test_data, e, left_on='date', right_on='date', how='outer')
#--------------------------------------------------------------------------#
# fit
#--------------------------------------------------------------------------#

	XTest = np.matrix(test_data.drop(['date'], axis=1))
	yHatTest = map(lr.predict, XTest) 
	# yHatTest = np.dot(XTest,w)
	print test_data.columns
#--------------------------------------------------------------------------#
#overwrite, output to csv
#--------------------------------------------------------------------------#
	temp = pd.read_csv("toyama.csv", header = None)
	for i in range(len(yHatTest)):
		temp.ix[i, (h + 1)] = yHatTest[i]
	temp.to_csv('toyama.csv', header = None, index = None)
#--------------------------------------------------------------------------#

temp = pd.read_csv("result.csv", header = None)
kita = pd.read_csv("toyama.csv", header = None)
kita.columns = ["date", "62", "63", "64"]
for i in range(len(temp)):
	temp.ix[i, 6] = 0.95 * (kita['62'][i] + kita['63'][i] + kita['64'][i])
temp.to_csv('result.csv', header = None, index = None)