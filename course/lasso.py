from sklearn import linear_model
import numpy as np
import pandas as pd

train_data = pd.read_csv("trainpp.csv")
train_data = train_data.drop(['kw_min_min'], axis=1) #remove 'n_non_stop_words' information.

X = np.matrix(train_data.drop(['shares'], axis=1)) 
y = np.array(train_data['shares'])

N = 18000

XTrain = X[:N,:]
yTrain = y[:N]
XVal = X[N:,:]
yVal = y[N:]

print "learn start"
#lr = linear_model.LinearRegression()
#lr = linear_model.ElasticNetCV()
#lr = linear_model.ElasticNet()
#lr = linear_model.LassoCV()
#lr = linear_model.Lasso()
lr = linear_model.RidgeCV()
#lr = linear_model.Ridge()
lr.fit(XTrain,yTrain)
print "learn end"

yHatTrain = map(lr.predict, XTrain)
yHatVal = map(lr.predict, XVal) 

resultTrain = np.mean(np.abs(yTrain - yHatTrain))
print "predict training end"
resultVal = np.mean(np.abs(yVal - yHatVal))
print "predict Validation end"

print "Training error ", resultTrain
print "Validation error ", resultVal

test_data = pd.read_csv("test.csv")
test_data = test_data.drop(['kw_min_min'], axis=1) #remove 'n_non_stop_words' information.
XTest = np.matrix(test_data)

yHatTest = map(lr.predict, XTest) 
print "predict Test end"
for i in range(9643):
	if yHatTest[i] > 1000000:
		yHatTest[i] = 1300
np.savetxt('result3.txt', yHatTest)