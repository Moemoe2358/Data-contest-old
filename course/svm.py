from sklearn import svm
#from sklearn import cross_validation
import numpy as np
import pandas as pd


train_data = pd.read_csv("trainppp.csv")
train_data = train_data.drop(['kw_min_min'], axis=1) #remove 'n_non_stop_words' information.

X = np.matrix(train_data.drop(['shares'], axis=1)) 

y = np.array(train_data['shares'])

N = 26646
XTrain = X[:N,:]
yTrain = y[:N]
XVal = X[N:,:]
yVal = y[N:]

# k = 0
# temp = 10000
# for i in range(4500,5000):
# 	N = i

# 	XTrain = X[:N,:] #use the first N samples for training
# 	yTrain = y[:N]

# 	reg = svm.SVR(kernel='poly', C=100000, degree=3).fit(XTrain, yTrain)
# 	yHatTrain = reg.predict(XTrain)
# 	resultTrain = np.mean(np.abs(yTrain - yHatTrain.T))
# 	#pa = float(600) / i - 0.02
# 	if resultTrain - temp > 0.1:
# 		k = k + 1
# 		print resultTrain, resultTrain - temp, i
# 		X = np.delete(X,(i - 1),axis=0)
# 		y = np.delete(y,(i - 1))
# 	temp = resultTrain

print "learn start"
reg = svm.SVR(kernel='rbf', C=500000).fit(XTrain, yTrain)
print "learn end"

yHatTrain = reg.predict(XTrain)
print yHatTrain
yHatVal = reg.predict(XVal)
print yHatVal

test_data = pd.read_csv("testppp.csv")
test_data = test_data.drop(['kw_min_min'], axis=1) #remove 'n_non_stop_words' information.

XTest = np.matrix(test_data)

yHatTest = reg.predict(XTest)
print yHatTest

np.savetxt('result.txt', yHatTest)

print "Training error ", np.mean(np.abs(yTrain - yHatTrain))
print "Vadlidation error ", np.mean(np.abs(yVal - yHatVal))