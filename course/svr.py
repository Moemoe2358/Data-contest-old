from sklearn import svm
import regression as reg
import numpy as np
import pandas as pd


train_data = pd.read_csv("trainppp.csv")
train_data = train_data.drop(['kw_min_min'], axis=1) #remove 'n_non_stop_words' information.

X = np.matrix(train_data.drop(['shares'], axis=1)) 

y = np.array(train_data['shares'])

temp = 1000
pa = 0.1
k = 0

for i in range(21000,26647):
	N = i

	XTrain = X[:N,:] #use the first N samples for training
	yTrain = y[:N]

	w = reg.ridgeRegres(XTrain,yTrain,3.89) #linear regression
	yHatTrain = np.dot(XTrain,w)
	resultTrain = np.mean(np.abs(yTrain - yHatTrain.T))
	#pa = float(600) / i - 0.02
	if resultTrain - temp > pa:
		k = k + 1
		print resultTrain, resultTrain - temp, i
		X = np.delete(X,(i - 1),axis=0)
		y = np.delete(y,(i - 1))
	temp = resultTrain

N = len(X) - 1
print N

XTrain = X[:N,:] #use the first N samples for training
yTrain = y[:N]
XVal = X[N:,:] #use the rests for validation
yVal = y[N:]

print "learn start"
reg = svm.SVR(kernel='rbf', C=100000).fit(XTrain, yTrain)
print "learn end"

yHatTrain = reg.predict(XTrain)
print yHatTrain
yHatVal = reg.predict(XVal)
print yHatVal

print "Training error ", np.mean(np.abs(yTrain - yHatTrain))
print "Validation error ", np.mean(np.abs(yVal - yHatVal))

test_data = pd.read_csv("testppp.csv")
test_data = test_data.drop(['kw_min_min'], axis=1) #remove 'n_non_stop_words' information.
XTest = np.matrix(test_data)
yHatTest = reg.predict(XTest)
print yHatTest
np.savetxt('result2.txt', yHatTest)