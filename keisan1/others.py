# -*- coding: utf-8 -*-

#--------------------------------------------------------------------------#
#0.部門0と1では、中央値は平均値よりいい。2と3は逆
#1.外れ値除去は効く
#2.訓練データチューニングは効く	
#3.モデルについて、一番効くのはridgeCV
#4.キーワードと人数の相関係数はあまり意味ない	

#benchmark           1.76417 1.64884
#1218 avg + col4     1.71097 1.40058
#1218 med + col4,5   1.64641 1.28595
#1218 avg + col4,5,8 1.67857 1.24936 
#1218 med + traindt  1.64783 1.29257
#1218 med + N330     1.65463 1.32429
#1219 col4,5,8       1.63857 1.24936
#1219 ??             1.74267 1.73517
#1219 sns            1.94149 2.66296
#1219 weather_wind   1.64063 1.25895
#1219 sensor_wet     1.64263 1.26828
#1220 outlier        1.62680 1.19441
#1220 outlier1.2     1.61829 1.15473
#1220 1.0            1.62007 1.16303
#1220 1.15 + dum     1.61961 1.16086
#1220 dum + weekend  1.61746 1.15084
#1221 skl + ridge0.2 1.61643 1.14602
#1221 ridge1.0       1.61595 1.14380
#1221 ridge3.0       1.61493 1.13903
#1221 ridgeCV        1.61274 1.12881
#1221 col3 + delbbs  1.57123 1.07756
#1222 col10 + del    1.55966 1.08057
#1222 col3 + add     1.54024 1.07643
#1222 col7           1.59168 1.07623
#1222 all            1.44643 1.07900
#1222 *1.1           1.54030 1.22286
#1223 all + syuku    1.44459 1.07939
#1223 syukuplus      1.44225 1.12116
#1223 N = 202        1.45763 1.09073
#1223 N = 272        1.44214 1.15256
#1223 N = 332        1.44462 1.22065
#1224 traintd        1.43490 1.14918
#1224 +kankou        1.49783 1.26755
#1224 del twitter    1.42428 1.06998
#1224 plus oturi     1.43166 1.10324
#1224 del hakone     1.42539 1.07520
#1225 col4,5,8entaka 1.45018 1.19118
#1225 log            1.41878 1.06483
#1225 keywordchange  1.43956 0.98639
#1225 col4,5,8       1.40197 0.98639
#1225 syuku          1.38448 0.99003
#1226 obonn          1.37948 1.01441
#1226 hoho           1.39351 1.03872
#1226 outlier        1.41250 1.07299
#1226 201409         1.38937 1.02102
#1226 keyword        1.41071 1.42975
#1227 keywordsum     1.43547 1.11085
#1227 all            1.43720 1.19698
#1227 col6,7         1.42975 1.01441
#1227 *1.5           1.44712 1.01441
#1227 mean*1.3       1.42965 1.60768
#1228 kawase         1.40264 1.00068
#1228 kawase4,5,8    1.37654 1.00068
#1228 kawase9,10     1.37654 1.00068
#1228 outlier1.1     1.37547 0.98820
#1228 N = 232        1.36966 0.96854
#1229 col6,7         1.37824 0.96854
#1229 del kawase,geo 1.37155 0.96854
#1229 keyword        1.36983 0.96854
#1229 del shine      1.37046 0.96854 1.26210
#1229 N = 310        1.37727 0.96854 1.30983
#1230 syukujitsu     1.38682 0.99666 1.30825
#1230 syukujitsu     1.37429 0.96854 1.28143
#1230 miseinenn      1.37000 0.96854 1.25892
#1230 event6         1.36827 0.96854 1.24684
#1230 N = 222, event 1.38637 0.96602 1.29044
#1231 shinkannsenn   1.39927 0.96602 1.38080
#1231 syukujitsu     1.37938 0.98117 1.38619
#1231 outliercancel  1.41097 1.04459 1.39040
#1231 outlierredo    1.36053 0.99440 1.23439
#1231 outlierreplace 1.39626 1.05854 1.24435
#0101 out1.2,4c      1.35760 0.98991 1.24255
#0101 linear1000, 3c 1.45725 1.06956 1.22822
#0101 rbf10000, N197 1.70152 1.10182 1.22822
#0101                1.35555 0.98991 1.22822
#0101 l10e4,1.1,0.97 1.41421 0.98117 1.22010
#0102 /40,,0.95      1.35255 0.96602 1.21700
#0102 scale,/30,0.9  1.44102 1.00219 1.22652
#0102 /50,/60,ev     1.35293 0.98069 1.22371
#0102 /50,usd,ev     1.34352 0.94311 1.21421
#0102                1.34587 0.94311 1.23342       
#0103                1.34655 0.94862 1.22719
#0103                1.33576 0.93541 
#0103 avg + syuku    1.32666 0.93541 1.10777 
#0103 del weekflag   1.33268 0.93541 1.14987
#0104 

#外部データ	
#季節性
#金沢キーワード、分ける
#特徴選択

##--------------------------------------------------------------------------#

#--------------------------------------------------------------------------#
#package
import numpy as np
import pandas as pd
import datetime
import csv
# import regression as reg
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
#--------------------------------------------------------------------------#
#build model for young person, mid-aged person, old person respectively
#--------------------------------------------------------------------------#
#自動化
pd.options.mode.chained_assignment = None
yid = ['14382_total', '14384_total', '22205_total', '04100_total', '26100_total', '13102_total', '01202_total', '24203_total',
 '32203_total', '34100_total', '42201_total', '47207_total', '16201_total', '17201_total']
colid = [4, 5, 8, 2, 10, 3, 1, 9, 11, 12, 13, 14, 6, 7]
weaid = ['w000008', 'w000008', 'w000011', 'w000003', 'w000023', 'w000006', 'w000001', 'w000022', 'w000026', 'w000029', 'w000032', 'w000035', 'w000017', 'w000013']
geoid = ['14382', '14384', '22205', '04100', '26100', '13102', '01202', '24203', '32203', '34100', '42201', '47207', '16201', '17201']
snsid = [
['date', '小田原_blog', '箱根_blog', '温泉_blog'],
['date', '温泉旅行_blog', '小田原_blog'],
['date', '釣り_blog', '熱海_blog', '温泉_blog'],
['date', '旅行_blog', '仙台_blog'],
['date', '旅行_blog', '京都_blog', '寺_blog', '嵐山_blog', '祇園_blog'],
['date', '旅行_blog', '観光_blog'],
['date', '旅行_blog', '函館_blog'],
['date', '旅行_blog', '伊勢_blog'],
['date', '旅行_blog', '出雲_blog'],
['date', '旅行_blog', '広島_blog'],
['date', '旅行_blog', '長崎_blog'],
['date', '旅行_blog', '石垣_blog'],
['date', '観光_blog'],
['date', '観光_blog', '金沢_blog']
]
#--------------------------------------------------------------------------#
# "ho" means the beginning and the end of the holidays
#--------------------------------------------------------------------------#
ho = ["2014-07-19", "2014-07-21", "2014-08-11", "2014-08-12"
, "2014-09-13",  "2014-09-15", "2014-09-20", "2014-09-23", "2014-10-11"
, "2014-10-13", "2014-11-01", "2014-11-03", "2014-11-22", "2014-11-24", "2014-12-27", "2014-12-28", "2014-12-29"
, "2014-12-30", "2015-01-03", "2015-01-04", "2015-01-10", "2015-01-12"
, "2015-05-02", "2015-07-18", "2015-07-20"
, "2015-08-10", "2015-08-11"
, "2015-09-19", "2015-09-23", "2015-10-10", "2015-10-12"
, "2015-11-21", "2015-11-23"]

#--------------------------------------------------------------------------#
# "hoho" means the middle of holidays
#--------------------------------------------------------------------------#
hoho = ["2014-07-20", "2014-08-13", "2014-08-14", "2014-08-15", "2014-09-14", "2014-09-21", "2014-09-22", "2014-10-12", "2014-11-02", "2014-11-23"
, "2014-12-31", "2015-01-01", "2015-01-02", "2015-01-11", "2015-05-03", "2015-05-04", "2015-05-05"
, "2015-07-19", "2015-08-12", "2015-08-13", "2015-08-14", "2015-09-20", "2015-09-21", "2015-09-22", "2015-10-11", "2015-11-22"]

# for h in range(len(yid)):
for h in range(3, 12):

#--------------------------------------------------------------------------#
#train_prepare
#--------------------------------------------------------------------------#

	train_all = pd.read_csv("target/target_train.csv")
	train_data = pd.DataFrame(train_all['date'])

#--------------------------------------------------------------------------#
#Add train weekdayflag,holidayflag, mid-holidayflag
# and turn them into dummy variable
#--------------------------------------------------------------------------#

	train_data = train_data.assign(weekday = 'NA')
	train_data = train_data.assign(hoho = 'NA')
	train_data = train_data.assign(holiday = 'NA')
	for i in range(len(train_data)):
		a = datetime.datetime.strptime(train_data['date'][i], "%Y-%m-%d")
		a = datetime.date(a.year, a.month, a.day)
		if str(a) in ho:
			train_data['holiday'][i] = 1
		else:
			train_data['holiday'][i] = 0
		if str(a) in hoho:
			train_data['hoho'][i] = 1
		else:
			train_data['hoho'][i] = 0
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

	geo = pd.read_csv("geo_location/geo_location_train.csv")
	c = geo[['date',geoid[h]]]
	train_data = pd.merge(train_data, c, left_on='date', right_on='date', how='outer')
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
	# for i in range(len(l)):
	# 	print l[i][1], l[i][0]
	d = sns[snsid[h]]

	for i in range(1, len(d)):
		d.iloc[:,i:i + 1] = np.log(d.iloc[:,i:i + 1])

	train_data = pd.merge(train_data, d, left_on='date', right_on='date', how='outer')
#--------------------------------------------------------------------------#

#--------------------------------------------------------------------------#
#train_sensor
# sen = pd.read_csv("sensor/sensor_train.csv")
# sen = sen[sen.ID == 33149937]
# e = sen[['date','wet0','wet6','wet12','wet18']]
# train_data = pd.merge(train_data, e, left_on='date', right_on='date', how='outer')

#--------------------------------------------------------------------------#
# pre-processing, delete outlier data, step 1:delete data outside thres
#--------------------------------------------------------------------------#

	X = np.matrix(train_data.drop(['date'], axis=1))
	y = np.matrix(train_all[yid[h]])

	Q1 = np.percentile(y, 75)
	Q3 = np.percentile(y, 25)
	thres = Q1 + 1.2 * (Q1 - Q3)

	k = 0
	pre = pd.concat((train_data, train_all[yid[h]]), axis=1)
	while k < len(pre):
		if pre[yid[h]][k] >= thres:
			# print pre['date'][k], pre[yid[h]][k]
			pre = pre.drop(pre.index[k])
			pre = pre.reset_index(drop=True)
			k = k - 1
		k = k + 1

	k = 0
	while k < len(X):
		if y[0, k] > thres:
			# y[0, k] = thres
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

	XVal = X
	yVal = y

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
			# print resultTrain, resultVal ,resultTrain - temp, k
			X = np.delete(X, (k - 1), axis=0)
			y = np.delete(y, (k - 1))
		else:
			temp = resultTrain
		k = k + 1

	N = 232
	XTrain = X[:N, :]
	yTrain = y[:, :N]
	XVal = X[N:, :]
	yVal = y[:, N:]
#--------------------------------------------------------------------------#
# Training fit, cross_validation
#--------------------------------------------------------------------------#
#train_evaluation
	#lr = linear_model.Ridge(alpha = 1)
	#lr = linear_model.LinearRegression()
	#lr = linear_model.ElasticNetCV()
	#lr = linear_model.ElasticNet()
	#lr = linear_model.LassoCV()
	#lr = linear_model.Lasso(alpha = 3)
	lr = linear_model.RidgeCV()
	lr = GradientBoostingRegressor()

	lr.fit(XTrain,np.squeeze(np.asarray(yTrain)))
	# print lr.coef_
	yHatTrain = map(lr.predict, XTrain)
	yHatVal = map(lr.predict, XVal)
	
	# w = reg.ridgeRegres(XTrain, yTrain, 0.2)
	# yHatTrain = np.dot(XTrain,w)
	# yHatVal = np.dot(XVal,w)
	y = np.squeeze(np.asarray(yTrain))
	# print colid[h], "Training error ", np.mean(np.abs(yTrain - yHatTrain.T))
	# print colid[h], "Validation error ", np.mean(np.abs(yVal - yHatVal.T))
	te = np.mean(np.abs(np.squeeze(np.asarray(yTrain)) - np.squeeze(np.asarray(yHatTrain))))
	ve = np.mean(np.abs(np.squeeze(np.asarray(yVal)) - np.squeeze(np.asarray(yHatVal))))
	print colid[h], "Training error ", te
	print colid[h], "Validation error ", ve, te / ve

  	kf = cross_validation.KFold(len(XTrain), n_folds=10)
  	error = 0
  	for training, test in kf:
  		lr.fit(XTrain[training], y[training])
  		p = np.array([lr.predict(xi) for xi in XTrain[test]])
  		e = np.squeeze(np.asarray(p)) - y[test]
  		error += np.mean(np.abs(e))
  	rmse_10cv = error / 10
	print colid[h], "CV error ", rmse_10cv

	lr.fit(XTrain,np.squeeze(np.asarray(yTrain)))
#--------------------------------------------------------------------------#
#test_prepare
#--------------------------------------------------------------------------#

	test_data = pd.read_csv("result.csv", header = None)
	temp = test_data
	test_data = test_data.iloc[:,0:1]
	test_data.columns = ["date"]
#--------------------------------------------------------------------------#
#Add test weekdayflag,holidayflag, mid-holidayflag
# and turn them into dummy variable
#--------------------------------------------------------------------------#
	test_data = test_data.assign(weekday = 'NA')
	test_data = test_data.assign(hoho = 'NA')
	test_data = test_data.assign(holiday = 'NA')
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
#test_geosns
#--------------------------------------------------------------------------#

	geot = pd.read_csv("geo_location/geo_location_test.csv")
	c = geot[['date',geoid[h]]]
	test_data = pd.merge(test_data, c, left_on='date', right_on='date', how='outer')
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
		meanVal = np.nanmedian(d.iloc[:, i:i + 1])
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
	# print test_data
#--------------------------------------------------------------------------#
#fit
	XTest = np.matrix(test_data.drop(['date'], axis=1))
	yHatTest = map(lr.predict, XTest) 
	# yHatTest = np.dot(XTest,w)
#--------------------------------------------------------------------------#
	# print test_data.columns
#--------------------------------------------------------------------------#
#overwrite
	for i in range(len(yHatTest)):
		temp.ix[i, colid[h]] = yHatTest[i]
	temp.to_csv('result.csv', header = None, index = None)
#--------------------------------------------------------------------------# 