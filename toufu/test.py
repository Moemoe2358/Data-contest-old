# -*- coding: utf-8 -*-
#holiday
#parameter chewing
#e?
import numpy as np
import pandas as pd
import datetime
import csv
from sklearn import linear_model
from sklearn import cross_validation

pd.options.mode.chained_assignment = None
yid = ['A', 'B', 'C', 'D', 'E']
sid = ['A_s', 'B_s', 'C_s', 'D_s', 'E_s']

for h in range(4, 5):
	train_all = pd.read_csv("target_train.csv")
	train_data = pd.DataFrame(train_all['date'])

	train_data = train_data.assign(weekday = 'NA')
	for i in range(len(train_data)):
		a = datetime.datetime.strptime(train_data['date'][i], "%Y-%m-%d")
		a = datetime.date(a.year, a.month, a.day)		
		train_data['weekday'][i] = a.isoweekday()
	dum = pd.get_dummies(train_data["weekday"])
	train_data = pd.concat((train_data, dum), axis = 1)
	train_data = train_data.drop(['weekday'], axis = 1)

	weather = pd.read_csv("weather.csv")
	if h == 4:
		weather = weather[weather.id == 'chichibu']
	# elif h == 1:
	# 	weather = weather[weather.id == 'tateyama']
	else:
		weather = weather[weather.id == 'tokyo']
	weather = weather[weather.date <= '2014-12-31']
	if h == 0 or h == 3:
		b = weather[['date','temperature','high_temperature','low_temperature','precipitation','hours_of_sunlight',
		'global_irradiance','deepest_snow','amount_of_snowfall','wind_velocity','humidity','atmospheric_pressure','degree_of_cloudiness']]
	else:
		b = weather[['date','temperature']]
	train_data = pd.merge(train_data, b, left_on = 'date', right_on = 'date', how = 'outer')

	sale = pd.read_csv("sale.csv")
	sale = sale[sale.date <= '2014-12-31']
	# c = sale[['date', sid[h]]]
	# train_data = pd.merge(train_data, c, left_on='date', right_on='date', how='outer')

	train_all[yid[h]] = train_all[yid[h]] - sale[sid[h]]


	X = np.matrix(train_data.drop(['date'], axis = 1))
	y = np.matrix(train_all[yid[h]])

	X = np.delete(X, 45, axis = 0)
	y = np.delete(y, 45)
	X = np.delete(X, 50, axis = 0)
	y = np.delete(y, 50)
	X = np.delete(X, 51, axis = 0)
	y = np.delete(y, 51)

	# k = 0
	# pre = pd.concat((train_data, train_all[yid[h]]), axis=1)
	# while k < len(pre):
	# 	if pre[yid[h]][k] >= thres:
	# 		print pre['date'][k], pre[yid[h]][k]
	# 		pre = pre.drop(pre.index[k])
	# 		pre = pre.reset_index(drop=True)
	# 		k = k - 1
	# 	k = k + 1

	Q1 = np.percentile(y, 75)
	Q3 = np.percentile(y, 25)
	thres = Q1 + 1.1 * (Q1 - Q3)

	k = 0
	while k < len(X):
		if y[0, k] >= thres or y[0, k] <= 0:
			X = np.delete(X, k, axis = 0)
			y = np.delete(y, k)
			k = k - 1
		k = k + 1

	lr = linear_model.RidgeCV()
	k = 100
	XTrain = X[:k, :]
	yTrain = y[:, :k]
	lr.fit(XTrain,np.squeeze(np.asarray(yTrain)))
	yHatTrain = map(lr.predict, XTrain)
	# yHatVal = map(lr.predict, XVal)
	temp = np.mean(np.abs(np.squeeze(np.asarray(yTrain)) - np.squeeze(np.asarray(yHatTrain))))
	while k < len(X):
		XTrain = X[:k, :]
		yTrain = y[:, :k]
		lr.fit(XTrain,np.squeeze(np.asarray(yTrain)))
		yHatTrain = map(lr.predict, XTrain)
		# yHatVal = map(lr.predict, XVal)
		resultTrain = np.mean(np.abs(np.squeeze(np.asarray(yTrain)) - np.squeeze(np.asarray(yHatTrain))))
		# resultVal = np.mean(np.abs(np.squeeze(np.asarray(yVal)) - np.squeeze(np.asarray(yHatVal))))
		if resultTrain - temp > resultTrain / 30:
			k = k - 1
			# print resultTrain, resultTrain - temp, k
			X = np.delete(X, (k - 1), axis = 0)
			y = np.delete(y, (k - 1))
		else:
			temp = resultTrain
		k = k + 1

	if h == 0 or h == 2 or h == 4:
		N = 241
		XTrain = X[:N, :]
		yTrain = y[:, :N]
		XVal = X[N:, :]
		yVal = y[:, N:]
	else:
		XTrain = X
		yTrain = y	
		XVal = X
		yVal = y

	#lr = linear_model.Ridge(alpha = 1)
	#lr = linear_model.LinearRegression()
	#lr = linear_model.ElasticNetCV()
	#lr = linear_model.ElasticNet()
	#lr = linear_model.LassoCV()
	#lr = linear_model.Lasso(alpha = 3)
	lr = linear_model.RidgeCV()

	lr.fit(XTrain,np.squeeze(np.asarray(yTrain)))
	# print lr.coef_
	yHatTrain = map(lr.predict, XTrain)
	yHatVal = map(lr.predict, XVal)

	# print colid[h], "Training error ", np.mean(np.abs(yTrain - yHatTrain.T))
	# print colid[h], "Validation error ", np.mean(np.abs(yVal - yHatVal.T))
	te = np.sqrt(np.mean(pow((np.squeeze(np.asarray(yTrain)) - np.squeeze(np.asarray(yHatTrain))), 2)))
	ve = np.sqrt(np.mean(pow((np.squeeze(np.asarray(yVal)) - np.squeeze(np.asarray(yHatVal))), 2)))
	print h, "Training error ", te
	print h, "Validation error ", ve

	test_data = pd.read_csv("sample_submit.csv", header = None)
	test_data = test_data.iloc[:, 0:1]
	test_data.columns = ["date"]

	test_data = test_data.assign(weekday = 'NA')
	for i in range(len(test_data)):
		a = datetime.datetime.strptime(test_data['date'][i], "%Y-%m-%d")
		a = datetime.date(a.year, a.month, a.day)		
		test_data['weekday'][i] = a.isoweekday()
	dum = pd.get_dummies(test_data["weekday"])
	test_data = pd.concat((test_data, dum), axis = 1)
	test_data = test_data.drop(['weekday'], axis = 1)
	
	weathert = pd.read_csv("weather.csv")
	if h == 4:
		weathert = weathert[weathert.id == 'chichibu']
	# elif h == 1:
	# 	weathert = weathert[weathert.id == 'tateyama']
	else:
		weathert = weathert[weathert.id == 'tokyo']
	weathert = weathert[weathert.date >= '2015-01-01']
	if h == 0 or h == 3:
		b = weathert[['date','temperature','high_temperature','low_temperature','precipitation','hours_of_sunlight',
		'global_irradiance','deepest_snow','amount_of_snowfall','wind_velocity','humidity','atmospheric_pressure','degree_of_cloudiness']]
	else:
		b = weathert[['date','temperature']]
	test_data = pd.merge(test_data, b, left_on = 'date', right_on = 'date', how = 'outer')

	salet = pd.read_csv("sale.csv")
	salet = salet[salet.date >= '2015-01-01']
	# c = salet[['date', sid[h]]]
	# test_data = pd.merge(test_data, c, left_on='date', right_on='date', how='outer')

	XTest = np.matrix(test_data.drop(['date'], axis = 1))
	yHatTest = map(lr.predict, XTest) 

	temp = pd.read_csv("sample_submit.csv", header = None)
	for i in range(len(yHatTest)):
		temp.ix[i, (h + 1)] = yHatTest[i] + salet.ix[i + 365, h + 1]
		if yHatTest[i] < 1 and i > 0:
			print "!"
			# print temp.ix[i, 0]
			temp.ix[i, (h + 1)] = temp.ix[i - 1, h + 1]
	temp.to_csv('sample_submit.csv', header = None, index = None)

	y = np.squeeze(np.asarray(yTrain))
  	kf = cross_validation.KFold(len(XTrain), n_folds=10)
  	error = 0
  	for training, test in kf:
  		lr.fit(XTrain[training], y[training])
  		p = np.array([lr.predict(xi) for xi in XTrain[test]])
  		e = np.squeeze(np.asarray(p)) - y[test]
  		error += pow(np.mean(np.abs(e)), 2)
  	rmse_10cv = np.sqrt(error / 10)
	print h, "CV error ", rmse_10cv
