import numpy as np
import pandas as pd
import regression as reg
import math

train_data = pd.read_csv("train.csv")
train_data = train_data.drop(['kw_min_min'], axis=1)
train_data = train_data.drop(['url'], axis=1)
train_data = train_data.drop(['timedelta'], axis=1)

X = np.matrix(train_data.drop(['shares'], axis=1)) 
y = np.matrix(train_data['shares']) #This is the target

XVal = X[0:,:] #use the rests for validation
yVal = y[:,0:]

s = 0
up = 4000
down = 10

while s < len(X):
	if y[0,s] > up or y[0,s] < down:
		X = np.delete(X,s,axis=0)
		y = np.delete(y,s)
		s -= 1
	s += 1
print len(X)

N = len(X) 
XTrain = X[:N,:] #use the first N samples for training
yTrain = y[:,:N]

w = reg.ridgeRegres(XTrain,np.log(yTrain),3.9)
# temp = 1000

# XVal = X[0:,:] #use the rests for validation
# yVal = y[:,0:]
# pa = 0.01
# k = 0

# for i in range(10000,26647):
# 	N = i

# 	XTrain = X[:N,:] #use the first N samples for training
# 	yTrain = y[:,:N]

# 	w = reg.ridgeRegres(XTrain,yTrain) #linear regression
# 	yHatTrain = np.dot(XTrain,w)
# 	resultTrain = np.mean(np.abs(yTrain - yHatTrain.T))
# 	if resultTrain - temp > pa:
# 		k = k + 1
# 		print resultTrain, resultTrain - temp, i
# 		X = np.delete(X,(i - 1),axis=0)
# 		y = np.delete(y,(i - 1))
# 	temp = resultTrain

yHatTrain = np.exp(np.dot(XTrain,w))
yHatVal = np.exp(np.dot(XVal,w))

print "Training error ", np.mean(np.abs(yTrain - yHatTrain.T))
print "Val error ", np.mean(np.abs(yVal - yHatVal.T))

test_data = pd.read_csv("test.csv")
test_data = test_data.drop(['kw_min_min'], axis=1)
yHatTest = np.exp(np.dot(np.matrix(test_data),w))
for i in range(9644):
	if yHatTest[i] < 0:
		yHatTest[i] = 1300
np.savetxt('result.txt', yHatTest)