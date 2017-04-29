#training data needed?
#automatic Fallback
#adjustment of length
import pandas as pd
import numpy as np
from sklearn import linear_model
import time

start_time = time.time()

data = pd.read_csv("testless.csv")
seqs = [pd.Series(x['Sequence'].split(',')) for _, x in data.iterrows()]
data['Last'] = [seq.value_counts(sort = False).idxmax() if seq.value_counts().max() != 1 else seq[len(seq) - 1] for seq in seqs]

num = -1
start = 0
end = 300
l = []

while start <= 113846:
	temp = data[start:end]
	seqs = [pd.Series(x['Sequence'].split(',')) for _, x in temp.iterrows()]
	for seq in seqs:
		num += 1
		if len(seq) < 10:
			continue

		XTrain, XVal = np.empty((0, 4), int), np.empty((0, 4), int)
		yTrain, yVal = np.array([]), np.array([])
		# for i in range(1, len(seq) / 2 - 1):
		# 	XTrain = np.append(XTrain, np.array([[int(seq[i]), int(seq[i + 1]), int(seq[i + 2]), int(seq[i + 3])]]), axis = 0)
		# 	yTrain = np.append(yTrain, np.array([int(seq[i + 4])]))
		for i in range(1, len(seq) - 4):
			XVal = np.append(XVal, np.array([[int(seq[i]), int(seq[i + 1]), int(seq[i + 2]), int(seq[i + 3])]]), axis = 0)
			yVal = np.append(yVal, np.array([int(seq[i + 4])]))
		lr = linear_model.LinearRegression()
		lr.fit(XVal, yVal)
		yHatVal = map(lr.predict, XVal)
		n = 0
		for i in range(len(yHatVal)):
			if yVal[i] == round(yHatVal[i]):
				n += 1
		accVal = n / float(len(yHatVal))
		XTest = map(lr.predict, np.array([[int(seq[len(seq) - 4]), int(seq[len(seq) - 3]), int(seq[len(seq) - 2]), int(seq[len(seq) - 1])]]))
		
		result = int(round(XTest[0]))
		highest = accVal
		choice = 4

		XTrain, XVal = np.empty((0, 3), int), np.empty((0, 3), int)
		yTrain, yVal = np.array([]), np.array([])
		# for i in range(1, len(seq) / 2 - 1):
		# 	XTrain = np.append(XTrain, np.array([[int(seq[i]), int(seq[i + 1]), int(seq[i + 2])]]), axis = 0)
		# 	yTrain = np.append(yTrain, np.array([int(seq[i + 3])]))
		for i in range(1, len(seq) - 3):
			XVal = np.append(XVal, np.array([[int(seq[i]), int(seq[i + 1]), int(seq[i + 2])]]), axis = 0)
			yVal = np.append(yVal, np.array([int(seq[i + 3])]))
		lr = linear_model.LinearRegression()
		lr.fit(XVal, yVal)
		yHatVal = map(lr.predict, XVal)
		n = 0
		for i in range(len(yHatVal)):
			if yVal[i] == round(yHatVal[i]):
				n += 1
		accVal = n / float(len(yHatVal))
		XTest = map(lr.predict, np.array([[int(seq[len(seq) - 3]), int(seq[len(seq) - 2]), int(seq[len(seq) - 1])]]))

		if accVal > highest:	
			result = int(round(XTest[0]))
			highest = accVal
			choice = 3

		XTrain, XVal = np.empty((0, 2), int), np.empty((0, 2), int)
		yTrain, yVal = np.array([]), np.array([])
		# for i in range(1, len(seq) / 2 - 1):
		# 	XTrain = np.append(XTrain, np.array([[int(seq[i]), int(seq[i + 1])]]), axis = 0)
		# 	yTrain = np.append(yTrain, np.array([int(seq[i + 2])]))
		for i in range(1, len(seq) - 2):
			XVal = np.append(XVal, np.array([[int(seq[i]), int(seq[i + 1])]]), axis = 0)
			yVal = np.append(yVal, np.array([int(seq[i + 2])]))
		lr = linear_model.LinearRegression()
		lr.fit(XVal, yVal)
		yHatVal = map(lr.predict, XVal)
		n = 0
		for i in range(len(yHatVal)):
			if yVal[i] == round(yHatVal[i]):
				n += 1
		accVal = n / float(len(yHatVal))
		XTest = map(lr.predict, np.array([[int(seq[len(seq) - 2]), int(seq[len(seq) - 1])]]))
		
		if accVal > highest:	
			result = int(round(XTest[0]))
			highest = accVal
			choice = 2

		l.append(highest)
		if highest > 0.1:
			data['Last'][num] = result
			print data['Id'][num], choice, highest, result

	start += 300
	end += 300

print np.mean(l)
data = data.drop(['Sequence'], axis = 1)
data.to_csv('submitless.csv', index = None)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"