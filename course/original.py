import numpy as np
import pandas as pd
import regression as reg

N = 20000

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data = train_data.drop(['kw_min_min'], axis=1) #remove 'n_non_stop_words' information.
test_data = test_data.drop(['kw_min_min'], axis=1) #remove 'n_non_stop_words' information.

X = np.matrix(train_data.drop(['shares'], axis=1)) 
y = np.matrix(train_data['shares']) #This is the target

XTrain = X[:N,:] #use the first N samples for training
yTrain = y[:,:N]
XVal = X[N:,:] #use the rests for validation
yVal = y[:,N:]

w = reg.standRegres(XTrain,yTrain) #linear regression

yHatTrain = np.dot(XTrain,w)
yHatVal = np.dot(XVal,w)

print "Training error ", np.mean(np.abs(yTrain - yHatTrain.T))
print "Validation error ", np.mean(np.abs(yVal - yHatVal.T))

yHatTest = np.dot(np.matrix(test_data),w)
np.savetxt('result.txt', yHatTest)
