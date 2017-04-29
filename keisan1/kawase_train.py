import numpy as np
import pandas as pd
import datetime
import csv

train = pd.read_csv("exchange/exchange_train.csv")
result = []
# result.append(['date,CAD_JPY,CNY_JPY,EUR_JPY,GBP_JPY,USD_JPY'])
d1 = datetime.date(2014, 6, 1)

while 1:
	t = train[train.date == str(d1)]
	t = t.reset_index()
	temp = []
	temp.append(d1)
	if len(t) > 0:
		for j in range(2, 7):
			temp.append(round(t.ix[0, j], 3))
	else:
		before = t
		b = -1
		while len(before) == 0 and b > -10:
			before = train[train.date == str(d1 + datetime.timedelta(days = b))]
			before = before.reset_index()
			b = b - 1
		after = t
		a = 1
		while len(after) == 0 and a < 10:
			after = train[train.date == str(d1 + datetime.timedelta(days = a))]
			after = after.reset_index()
			a = a + 1
		if b == -10:
			before = after
		if a == 10:
			after = before
		for j in range(2, 7):
			temp.append(round((before.ix[0, j] + after.ix[0, j]) / 2.0, 3))

	result.append(temp)
	if str(d1) == "2015-05-31":
		break
	d1 = d1 + datetime.timedelta(days = 1)

df = pd.DataFrame(result)
df.columns = ["date","CAD_JPY","CNY_JPY","EUR_JPY","GBP_JPY","USD_JPY"]
df.to_csv('exchange/kawase_train.csv', index = None)