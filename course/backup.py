import numpy as np
import pandas as pd
import regression as reg

#N = 23244
N = 26610

train_data = pd.read_csv("trainpp.csv")

train_data = train_data.drop(['url'], axis=1) #remove 'url' information.
train_data = train_data.drop(['timedelta'], axis=1) #remove 'timedelta' information.
# train_data = train_data.drop(['global_rate_positive_words'], axis=1) #remove 'weekday_is_monday' information.
# train_data = train_data.drop(['data_channel_is_socmed'], axis=1) #remove 'weekday_is_tuesday' information.
# train_data = train_data.drop(['abs_title_subjectivity'], axis=1) #remove 'weekday_is_wednesday' information.
# train_data = train_data.drop(['kw_min_max'], axis=1) #remove 'weekday_is_thursday' information.
train_data = train_data.drop(['global_sentiment_polarity'], axis=1) #remove 'weekday_is_friday' information.
train_data = train_data.drop(['n_unique_tokens'], axis=1) #remove 'weekday_is_saturday' information.
train_data = train_data.drop(['n_non_stop_words'], axis=1) #remove 'n_non_stop_words' information.
train_data = train_data.drop(['min_positive_polarity'], axis=1) #remove 'n_non_stop_words' information.
train_data = train_data.drop(['LDA_00'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['weekday_is_wednesday'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['num_self_hrefs'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['average_token_length'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['weekday_is_friday'], axis=1) #remove 'n_non_stop_words' information.

X = np.matrix(train_data.drop(['shares'], axis=1)) 
y = np.matrix(train_data['shares']) #This is the target

yMean = np.mean(y[:,:30000])
yMedian = np.median(y[:,:30000], axis=1)
yMax = np.max(y[:,:30000])
yMin = np.min(y[:,:30000])
print yMean,yMedian,yMax,yMin

X = np.delete(X,(17923,18072,18118,18213,18240,18328,18587,18858,19762,20161,20172,20443,20648,20800,20951,21122,21601,22051,22159,23007,23186,24512,24895,24967,25069,25073,25102,25133,25234,25260,25323,25608,25820,25902,26055,26333,26440),axis=0)
y = np.delete(y,(17923,18072,18118,18213,18240,18328,18587,18858,19762,20161,20172,20443,20648,20800,20951,21122,21601,22051,22159,23007,23186,24512,24895,24967,25069,25073,25102,25133,25234,25260,25323,25608,25820,25902,26055,26333,26440))

XTrain = X[:N,:] #use the first N samples for training
yTrain = y[:,:N]

XVal = X[0:,:] #use the rests for validation
yVal = y[:,0:]

w = reg.ridgeRegres(XTrain,yTrain,3) #linear regression

yHatTrain = np.dot(XTrain,w)
yHatVal = np.dot(XVal,w)

for i in range(N):
	if yHatTrain[i] < 0:
		yHatTrain[i] = yMedian

for i in range(len(XVal)):
	if yHatVal[i] < 0:
		yHatVal[i] = yMedian

print "Training error ", np.mean(np.abs(yTrain - yHatTrain.T))
print "Validation error ", np.mean(np.abs(yVal - yHatVal.T))


test_data = pd.read_csv("test.csv")
# test_data = test_data.drop(['global_rate_positive_words'], axis=1) #remove 'weekday_is_monday' information.
# test_data = test_data.drop(['data_channel_is_socmed'], axis=1) #remove 'weekday_is_tuesday' information.
# test_data = test_data.drop(['abs_title_subjectivity'], axis=1) #remove 'weekday_is_wednesday' information.
# test_data = test_data.drop(['kw_min_max'], axis=1) #remove 'weekday_is_thursday' information.
test_data = test_data.drop(['global_sentiment_polarity'], axis=1) #remove 'weekday_is_friday' information.
test_data = test_data.drop(['n_unique_tokens'], axis=1) #remove 'weekday_is_saturday' information.
test_data = test_data.drop(['n_non_stop_words'], axis=1) #remove 'n_non_stop_words' information.
test_data = test_data.drop(['min_positive_polarity'], axis=1) #remove 'n_non_stop_words' information.
test_data = test_data.drop(['LDA_00'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['weekday_is_wednesday'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['num_self_hrefs'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['average_token_length'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['weekday_is_friday'], axis=1) #remove 'n_non_stop_words' information.

yHatTest = np.dot(np.matrix(test_data),w)
for i in range(9643):
	if yHatTest[i] < 0:
		yHatTest[i] = yMedian
np.savetxt('result.txt', yHatTest)
