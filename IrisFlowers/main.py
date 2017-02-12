import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_idx = [0, 50, 100]

setosa = 0
versicolor = 1
virginica = 2

# training data
training_target = np.delete(iris.target, test_idx)
training_data = np.delete(iris.data, test_idx, axis=0)

# test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# print iris.feature_names
# print iris.target_names
# print iris.data[0]
# print iris.target[0]

clf = tree.DecisionTreeClassifier().fit(training_data, training_target)
print test_target # result [0, 1, 2]
print clf.predict(test_data) # result [0, 1, 2]

# https://www.youtube.com/watch?v=tNa99PG8hR8

