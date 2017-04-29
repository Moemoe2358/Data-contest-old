import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("train.csv")

sub1 = train_data[train_data.shares > 5800]
sub0 = train_data[train_data.shares <= 5800]

temp1 = pd.DataFrame([1] * len(sub1), index = sub1.index)
temp0 = pd.DataFrame([0] * len(sub0), index = sub0.index)

sub1['flag'] = temp1
sub0['flag'] = temp0

sub = sub1.append(sub0, ignore_index=True)
sub = sub.iloc[np.random.permutation(len(sub))]

sub = sub.drop(['url'], axis=1) #remove 'url' information.
sub = sub.drop(['timedelta'], axis=1) #remove 'url' information.
sub = sub.drop(['shares'], axis=1) #remove 'url' information.
X = np.matrix(sub.drop(['flag'], axis=1)) 
y = np.array(sub['flag']) #This is the target
#print sub
N = 20000
XTrain = X[:N,:]
yTrain = y[:N]
XVal = X[N:,:]
yVal = y[N:]

model = RandomForestClassifier(n_estimators = 5)
model.fit(XTrain, yTrain)
outputTrain = model.predict(XTrain)
outputVal = model.predict(XVal)

#print output[1],y[1]
TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(outputTrain)):
	if outputTrain[i] == 1 and yTrain[i] == 1:
		TP = TP + 1
	if outputTrain[i] == 1 and yTrain[i] == 0:
		FP = FP + 1 
	if outputTrain[i] == 0 and yTrain[i] == 0:
		TN = TN + 1 
	if outputTrain[i] == 0 and yTrain[i] == 1:
		FN = FN + 1 

print TP,FN
print FP,TN
print float(TP) / (TP + FP)

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(outputVal)):
	if outputVal[i] == 1 and yVal[i] == 1:
		TP = TP + 1
	if outputVal[i] == 1 and yVal[i] == 0:
		FP = FP + 1 
	if outputVal[i] == 0 and yVal[i] == 0:
		TN = TN + 1 
	if outputVal[i] == 0 and yVal[i] == 1:
		FN = FN + 1 

print TP,FN
print FP,TN
print float(TP) / (TP + FP)

#sub0 = pd.concat([sub0,temp0])
# sub = train_data.query('shares < 5900')

# df = DataFrame({'gender': np.random.choice(['m', 'f'], size=10), 'price': poisson(100, size=10)})

# df.query('gender == "m" and price < 100')

# X = np.matrix(train_data.drop(['shares'], axis=1)) 
# y = np.array(train_data['shares']) #This is the target

#print sub1

# for i in range(30000):
# 	if y[i] > 5800:
# 		X = np.delete(X,(i - 1),axis=0)
# 		y = np.delete(y,(i - 1))
# 		X[i].append(2)

