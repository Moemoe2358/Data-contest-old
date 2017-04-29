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
		highest = 0
		choice = 0

		for h in reversed(range(2, 13)):

			if len(seq) < (h + 3):
				continue

			XTrain, XVal = np.empty((0, h), int), np.empty((0, h), int)
			yTrain, yVal = np.array([]), np.array([])

			for i in range(1, len(seq) - h):

				templ = []
				for j in range(h):
					templ.append(int(seq[i + j]))
				XVal = np.append(XVal, np.array([templ]), axis = 0)

				yVal = np.append(yVal, np.array([int(seq[i + h])]))
		
			lr = linear_model.LinearRegression()

			lr.fit(XVal, yVal)
			yHatVal = map(lr.predict, XVal)
			n = 0
			for i in range(len(yHatVal)):
				if yVal[i] == round(yHatVal[i]):
					n += 1
			accVal = n / float(len(yHatVal))
		
			if accVal > highest and accVal > 0.1:

				templ = []
				for j in reversed(range(h)):
					templ.append(int(seq[len(seq) - j - 1]))
				result = int(round(map(lr.predict, np.array([templ]))[0]))

				highest = accVal
				choice = h

				if accVal == 1:
					break

		l.append(highest)

		if choice != 0:
			print data['Id'][num], choice, highest, result
			data['Last'][num] = result

	start += 300
	end += 300

print np.mean(l)

data = data.drop(['Sequence'], axis = 1)
data.to_csv('submit.csv', index = None)

elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"