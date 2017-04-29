#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *
from pandas import *
from sklearn import svm 
N=600


df = read_csv("train.csv")# use pandas to read  csv
df=df.drop(["PassengerId","Name","Ticket","Cabin","Embarked",'Age','Fare','Pclass','SibSp'],axis=1)# delete variables we do not want


df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)# creat dummy variable to describe sex
df_new = df

# convert dataframe into matrix
x = np.matrix(df_new.drop(['Survived'], axis=1)) 
y = np.matrix(df_new['Survived']) #This is the target

XTrain = x[:N,:] #use the first N samples for training
yTrain = y[:,:N].T

XVal = x[N:,:] #use the rests for validation
yVal = y[:,N:].T


clf = svm.SVC(kernel='rbf') #Support Vector Classification,RBFカーネルを使用
clf.fit(XTrain,yTrain) #学習
pre_list = clf.predict(XVal) #予測

# evaluate outcome
pre_num = len(yVal)
num_answer = 0
for i in range(pre_num):
    if yVal[i] == pre_list[i]:
        num_answer += 1   #分類結果を表示する
accuracy = (num_answer*1.0/pre_num)*100
print accuracy
