import numpy as np
import pandas as pd
import regression as reg

t1 = pd.read_csv("resultmoemoe.txt", header = None)
t2 = pd.read_csv("resultChilly.txt", header = None)

m1 = np.matrix(t1)
m2 = np.matrix(t2)
r = (m1 + m2) / 2
print m1, m2, r
np.savetxt('resultmoeChilly.txt', r)