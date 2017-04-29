import numpy as np
import pandas as pd
import regression as reg

minTrain = 3000000
minVal = 3000000
markTrain = 0
markVal = 0

train_data = pd.read_csv("trainpp.csv")
# train_data = train_data.drop(['url'], axis=1) #remove 'url' information.
# train_data = train_data.drop(['timedelta'], axis=1) #remove 'timedelta' information.
# train_data = train_data.drop(['global_rate_positive_words'], axis=1) #remove 'weekday_is_monday' information.
# train_data = train_data.drop(['data_channel_is_socmed'], axis=1) #remove 'weekday_is_tuesday' information.
# train_data = train_data.drop(['abs_title_subjectivity'], axis=1) #remove 'weekday_is_wednesday' information.
# train_data = train_data.drop(['kw_min_max'], axis=1) #remove 'weekday_is_thursday' information.
# train_data = train_data.drop(['global_sentiment_polarity'], axis=1) #remove 'weekday_is_friday' information.
# train_data = train_data.drop(['n_unique_tokens'], axis=1) #remove 'weekday_is_saturday' information.
# train_data = train_data.drop(['n_non_stop_words'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['min_positive_polarity'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['LDA_00'], axis=1) #remove 'n_non_stop_words' information.
train_data = train_data.drop(['kw_min_min'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['weekday_is_saturday'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['weekday_is_sunday'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['weekday_is_tuesday'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['weekday_is_monday'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['weekday_is_wednesday'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['weekday_is_friday'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['weekday_is_thursday'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['num_self_hrefs'], axis=1) #remove 'n_non_stop_words' information.
# train_data = train_data.drop(['average_token_length'], axis=1) #remove 'n_non_stop_words' information.


X = np.matrix(train_data.drop(['shares'], axis=1)) 
y = np.matrix(train_data['shares']) #This is the target

yMean = np.mean(y[:,:30000])
yMedian = np.median(y[:,:30000], axis=1)
yMax = np.max(y[:,:30000])
yMin = np.min(y[:,:30000])

XTrain = X[:2999,:] #use the first N samples for training
yTrain = y[:,:2999]

w = reg.ridgeRegres(XTrain,yTrain,3.89) #linear regression
yHatTrain = np.dot(XTrain,w)
resultTrain = np.mean(np.abs(yTrain - yHatTrain.T))
temp = resultTrain

XVal = X[0:,:] #use the rests for validation
yVal = y[:,0:]
pa = 0.1
k = 0
for i in range(3000,26647):
	XTrain = X[:i,:] #use the first N samples for training
	yTrain = y[:,:i]

	w = reg.ridgeRegres(XTrain,yTrain,3.89) #linear regression
	yHatTrain = np.dot(XTrain,w)
	yHatVal = np.dot(XVal,w)
	resultTrain = np.mean(np.abs(yTrain - yHatTrain.T))
	resultVal = np.mean(np.abs(yVal - yHatVal.T))
	#pa = float(600) / i - 0.02
	if resultTrain - temp > pa:
		k = k + 1
		print resultTrain, resultVal ,resultTrain - temp, i
		X = np.delete(X,(i - 1),axis=0)
		y = np.delete(y,(i - 1))
	else:
		temp = resultTrain

	# if resultTrain < minTrain:
	# 	minTrain = resultTrain
	# 	markTrain = i
	# if resultVal < minVal:
	# 	minVal = resultVal
	# 	markVal = i
alpha = 0.0
bestTrain = 0
bestVal = 0

for i in range(1000):
	w = reg.ridgeRegres(XTrain,yTrain,alpha) #linear regression	
	yHatTrain = np.dot(XTrain,w)
	yHatVal = np.dot(XVal,w)
	resultTrain = np.mean(np.abs(yTrain - yHatTrain.T))
	resultVal = np.mean(np.abs(yVal - yHatVal.T))
	if resultTrain < minTrain:
		minTrain = resultTrain
		markTrain = i
		bestTrain = alpha
	if resultVal < minVal:
		minVal = resultVal
		markVal = i
		bestVal = alpha
	print alpha, resultTrain, resultVal
	alpha = alpha + 0.01

print bestTrain,bestVal

for i in range(len(XTrain)):
	if yHatTrain[i] < 0:
		yHatTrain[i] = yMedian

for i in range(len(XVal)):
	if yHatVal[i] < 0:
		yHatVal[i] = yMedian

w = reg.ridgeRegres(XTrain,yTrain,bestTrain)
yHatTrain = np.dot(XTrain,w)
yHatVal = np.dot(XVal,w)

resultTrain = np.mean(np.abs(yTrain - yHatTrain.T))
resultVal = np.mean(np.abs(yVal - yHatVal.T))

print "Training error ", resultTrain
print "Validation error ", resultVal
print "k = ", k

test_data = pd.read_csv("test.csv")
# test_data = test_data.drop(['global_sentiment_polarity'], axis=1) #remove 'weekday_is_friday' information.
# test_data = test_data.drop(['n_unique_tokens'], axis=1) #remove 'weekday_is_saturday' information.
# test_data = test_data.drop(['n_non_stop_words'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['min_positive_polarity'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['LDA_00'], axis=1) #remove 'n_non_stop_words' information.
test_data = test_data.drop(['kw_min_min'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['weekday_is_saturday'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['weekday_is_sunday'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['weekday_is_tuesday'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['weekday_is_monday'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['weekday_is_wednesday'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['weekday_is_friday'], axis=1) #remove 'n_non_stop_words' information.
# test_data = test_data.drop(['weekday_is_thursday'], axis=1) #remove 'n_non_stop_words' information.
yHatTest = np.dot(np.matrix(test_data),w)

for i in range(9644):
	if yHatTest[i] < 0:
		yHatTest[i] = yMedian
np.savetxt('result.txt', yHatTest)
