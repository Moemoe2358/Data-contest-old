from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import numpy as np

data = load_iris()

features = data['data']
feature_names = data['feature_names']
target = data['target']
target_names = data['target_names']
labels = target_names[target]

# for t,marker,c in zip(xrange(3),">ox","rgb"):
# 	plt.scatter(features[target == t,0],
# 		features[target == t,1],
# 		marker=marker,
# 		c=c,
# 		label=target_names[t])
# 	plt.legend()

# plt.xlabel(feature_names[0])
# plt.ylabel(feature_names[1])

is_setosa = (labels == 'setosa')

features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')

best_acc = -1.0

for fi in range(features.shape[1]):
    thresh = features[:, fi].copy()
    thresh.sort()
    for t in thresh:
        pred = (features[:, fi] > t)
        acc = (np.sum(labels[pred] == 'virginica') + np.sum(labels[~pred] != 'virginica')) / float(100)

        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t

print best_t, data.feature_names[best_fi], best_fi, best_acc

plt.show()
