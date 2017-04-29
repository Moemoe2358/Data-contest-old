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
cid = ['abiko',
'chiba',
'chichibu',
'choshi',
'ebina',
'edogawarinkai',
'fuchuu',
'funabashi',
'hachioji',
'hakone',
'haneda',
'hanno',
'hatoyama',
'hiratsuka',
'hiyoshi',
'kamiyoshida',
'kamogawa',
'katori',
'katsuura',
'kisarazu',
'konosu',
'koshigaya',
'kuki',
'kumagaya',
'kyonan',
'mitsumine',
'miura',
'mobara',
'narita',
'nerima',
'odawara',
'ogouchi',
'ome',
'otaki',
'ozawa',
'sagamiharachuo',
'sagamiko',
'saitama',
'sakahata',
'sakura',
'setagaya',
'tanzawako',
'tateyama',
'tokigawa',
'tokorozawa',
'tokyo',
'tonosho',
'tsujido',
'urayama',
'ushiku',
'yokohama',
'yokoshibahikari'
]

for h in range(1, 2):

	for hh in range(len(cid)):
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
		weather = weather[weather.id == cid[hh]]
		weather = weather.reset_index(drop = True)
		if pd.isnull(weather.ix[1, 2]):
			print cid[hh], "!"
			continue
		weather = weather[weather.date <= '2014-12-31']
		# b = weather[['date','temperature']]
		b = weather[['date','temperature','high_temperature','low_temperature','precipitation']]
		
		numFeat = np.shape(b)[1]
		for i in range(1, numFeat):
			medVal = np.median(b.iloc[:, i:i + 1])
			b.iloc[:,i:i + 1] = b.fillna(medVal)

		train_data = pd.merge(train_data, b, left_on = 'date', right_on = 'date', how = 'outer')

		sale = pd.read_csv("sale.csv")
		sale = sale[sale.date <= '2014-12-31']

		train_all[yid[h]] = train_all[yid[h]] - sale[sid[h]]

		X = np.matrix(train_data.drop(['date'], axis = 1))
		y = np.matrix(train_all[yid[h]])

		X = np.delete(X, 45, axis = 0)
		y = np.delete(y, 45)
		X = np.delete(X, 50, axis = 0)
		y = np.delete(y, 50)
		X = np.delete(X, 51, axis = 0)
		y = np.delete(y, 51)

		Q1 = np.percentile(y, 75)
		Q3 = np.percentile(y, 25)
		thres = Q1 + 1.2 * (Q1 - Q3)

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
		temp = np.mean(np.abs(np.squeeze(np.asarray(yTrain)) - np.squeeze(np.asarray(yHatTrain))))
		while k < len(X):
			XTrain = X[:k, :]
			yTrain = y[:, :k]
			lr.fit(XTrain,np.squeeze(np.asarray(yTrain)))
			yHatTrain = map(lr.predict, XTrain)
			resultTrain = np.mean(np.abs(np.squeeze(np.asarray(yTrain)) - np.squeeze(np.asarray(yHatTrain))))
			if resultTrain - temp > resultTrain / 30:
				k = k - 1
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

		lr = linear_model.RidgeCV()

		lr.fit(XTrain,np.squeeze(np.asarray(yTrain)))
		yHatTrain = map(lr.predict, XTrain)
		yHatVal = map(lr.predict, XVal)

		te = np.sqrt(np.mean(pow((np.squeeze(np.asarray(yTrain)) - np.squeeze(np.asarray(yHatTrain))), 2)))
		ve = np.sqrt(np.mean(pow((np.squeeze(np.asarray(yVal)) - np.squeeze(np.asarray(yHatVal))), 2)))
		print cid[hh], "Training error ", te
		print cid[hh], "Validation error ", ve

		y = np.squeeze(np.asarray(yTrain))
	  	kf = cross_validation.KFold(len(XTrain), n_folds=10)
	  	error = 0
	  	for training, test in kf:
	  		lr.fit(XTrain[training], y[training])
	  		p = np.array([lr.predict(xi) for xi in XTrain[test]])
	  		e = np.squeeze(np.asarray(p)) - y[test]
	  		error += pow(np.mean(np.abs(e)), 2)
	  	rmse_10cv = np.sqrt(error / 10)
		print cid[hh], "CV error ", rmse_10cv
		print ""