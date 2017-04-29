# # -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('report.csv', skiprows = 5, header = None, nrows = 630)
df.index = pd.date_range('2004-01-04', periods = 630, freq = 'W')
del df[0]
df = df.rename_axis(mapper = {1:'num'}, axis=1)
dfm = df.resample('M', how=sum)
# dfm.plot()

dfms = dfm[dfm.index <= '2013-12-31']
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(dfms['num'], lags=20, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(dfms['num'], lags=20, ax=ax2)
# fig.show()

dfms.num = dfms.num.astype(float)
arma_mod = sm.tsa.ARMA(dfms, (11, 1)).fit()
# arma_mod1200 = sm.tsa.ARMA(dfms, (12,0)).fit()
print arma_mod.aic
resid = arma_mod.resid
# resid.plot()

# fig = plt.figure(figsize = (12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags = 20, ax = ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(resid, lags = 20, ax = ax2)
# fig.show()

# print stats.normaltest(resid)

# r, q, p = sm.tsa.acf(resid.values.squeeze(), nlags = 13, qstat = True)
# data = np.c_[range(1, 14), r[1:], q, p]
# table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print table.set_index('lag')

dfm.plot()
pr = arma_mod.predict(119, 143, dynamic = True)
pr.plot(style = 'r--')

print pr
s = 0
for i in range(24):
	s += dfm['num'][119 + i] - pr[i]

print s

plt.show()
