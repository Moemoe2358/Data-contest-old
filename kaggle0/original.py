import pandas as pd
import numpy as np
import csv as csv
from sklearn.cross_validation import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
pd.options.mode.chained_assignment = None

# TRAIN DATA
train_df = pd.read_csv('train.csv', header = 0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.
# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values
Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Embarked','Fare','Parch','SibSp','Pclass'], axis=1) 

print train_df
# TEST DATA
test_df = pd.read_csv('test.csv', header = 0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Embarked','Fare','Parch','SibSp','Pclass'], axis=1) 


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

# forest = RandomForestClassifier(n_estimators = 100)
# forest = GaussianNB()
# forest = tree.DecisionTreeClassifier()
# forest = svm.SVC(kernel='rbf')
# forest = neighbors.KNeighborsClassifier(n_neighbors = 2)
forest = LogisticRegression()

scores = []
cv = KFold(n = len(train_data), n_folds = 10, shuffle = True)

for train, test in cv:
	X_train, y_train = train_data[train, 1::], train_data[train, 0]
	X_test, y_test = train_data[test, 1::], train_data[test, 0]
	forest = forest.fit(X_train, y_train)
	scores.append(forest.score(X_test, y_test))

print "CV", np.mean(scores)

print 'Training...'
forest = forest.fit(train_data[::,1::], train_data[::,0])

print 'Predicting...'
output = forest.predict(test_data).astype(int)
predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

# outputTrain = forest.predict(train_data[:N,1::])
# TP = 0
# TN = 0
# FP = 0
# FN = 0
# for i in range(N):
# 	if outputTrain[i] == 1 and train_data[i,0] == 1:
# 		TP = TP + 1
# 	if outputTrain[i] == 1 and train_data[i,0] == 0:
# 		FP = FP + 1 
# 	if outputTrain[i] == 0 and train_data[i,0] == 0:
# 		TN = TN + 1 
# 	if outputTrain[i] == 0 and train_data[i,0] == 1:
# 		FN = FN + 1 

# print TP,FN
# print FP,TN
# print float(TP) / (TP + FP)