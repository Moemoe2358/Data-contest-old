import numpy as np
import pandas as pd

t1 = pd.read_csv("result1.csv", header = None, index_col = 0)
t2 = pd.read_csv("result2.csv", header = None, index_col = 0)

t = (t1 + t2) / 2

t.to_csv("result.csv", header = None)
# m1 = np.matrix(t1)
# m2 = np.matrix(t2)
# r = (m1 + m2) / 2
# # print m1, m2, r
# np.savetxt('result.csv', r)
print t