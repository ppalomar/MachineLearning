from sklearn import tree

orange = 1
apple = 0
smooth = 0
bumpy = 1

features = [[140, smooth], [130, smooth], [150, bumpy], [170, bumpy]]
fruits = [apple, apple, orange, orange]
clf = tree.DecisionTreeClassifier().fit(features, fruits)

print 'orange' if clf.predict([200, bumpy]) == orange else 'apple'
