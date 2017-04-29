# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# import csv as csv
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

train_df = pd.read_csv('train.csv', header = 0) #訓練データを読み込む

train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int) #性別という特徴の前処理

#年齢という特徴の前処理（欠損値補完）
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# dum = pd.get_dummies(train_df["Pclass"])
# train_df = pd.concat((train_df, dum), axis = 1)

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Parch', 'SibSp', 'Fare', 'Pclass'], axis = 1) #不要な特徴を除去
train_data = train_df.values #マトリックスに転換

pd.set_option('display.width', 1000)
# print train_df.head(10)
# model = svm.SVC(kernel = 'rbf') #モデル選択
# model = RandomForestClassifier(n_estimators = 100)
# model = GaussianNB()
# model = neighbors.KNeighborsClassifier(n_neighbors = 2)
# model = LogisticRegression()
model = tree.DecisionTreeClassifier()

#交差検定
scores = []
cv = KFold(n = len(train_data), n_folds = 10, shuffle = True)
for train, vaildation in cv:
	X_train, y_train = train_data[train, 1::], train_data[train, 0]
	X_vaildation, y_vaildation = train_data[vaildation, 1::], train_data[vaildation, 0]
	model.fit(X_train, y_train) #学習
	scores.append(model.score(X_vaildation, y_vaildation)) #予測、精度を計算

outputTrain = model.predict(X_train)
outputVaildation = model.predict(X_vaildation)

print "CV", np.mean(scores)
TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(outputTrain)):
	if outputTrain[i] == 1 and y_train[i] == 1:
		TP = TP + 1
	if outputTrain[i] == 1 and y_train[i] == 0:
		FP = FP + 1 
	if outputTrain[i] == 0 and y_train[i] == 0:
		TN = TN + 1 
	if outputTrain[i] == 0 and y_train[i] == 1:
		FN = FN + 1 
print "Confusion Matrix:"
print TP,FN
print FP,TN
print "Accuracy", float(TP) / (TP + FP)
print ""

outputVail = model.predict(X_vaildation)
TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(outputVail)):
	if outputVail[i] == 1 and y_vaildation[i] == 1:
		TP = TP + 1
	if outputVail[i] == 1 and y_vaildation[i] == 0:
		FP = FP + 1 
	if outputVail[i] == 0 and y_vaildation[i] == 0:
		TN = TN + 1 
	if outputVail[i] == 0 and y_vaildation[i] == 1:
		FN = FN + 1 
print "Confusion Matrix:"
print TP,FN
print FP,TN
print "Accuracy", float(TP) / (TP + FP)
print ""

#kaggleに提出したい場合、テストの部分も
# test_df = pd.read_csv('test.csv', header = 0)
# test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# median_age = test_df['Age'].dropna().median()
# if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
#     test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# ids = test_df['PassengerId'].values #IDをあらかじめ記録する
# test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Parch', 'Fare', 'SibSp', 'Pclass'], axis = 1)
# test_data = test_df.values

# forest = forest.fit(train_data[::,1::], train_data[::,0]) #学習
# output = forest.predict(test_data).astype(int) #予測
# #提出ための手続き
# predictions_file = open("myfirstforest.csv", "wb")
# open_file_object = csv.writer(predictions_file)
# open_file_object.writerow(["PassengerId", "Survived"])
# open_file_object.writerows(zip(ids, output))
# predictions_file.close()