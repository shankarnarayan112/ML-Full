from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
     [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Classifiers
clf_tree = DecisionTreeClassifier()
clf_svc = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()
clf_GaussianNB = GaussianNB()

# Training
clf_svc = clf_svc.fit(X,Y)
clf_KNN = clf_KNN.fit(X,Y)
clf_tree = clf_tree.fit(X,Y)
clf_perceptron = clf_perceptron.fit(X,Y)
clf_GaussianNB = clf_GaussianNB.fit(X,Y)

# Testing (using the same data-set)
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svc = clf_svc.predict(X)
acc_svc = accuracy_score(Y, pred_svc) * 100
print('Accuracy for SVM: {}'.format(acc_svc))

pred_per = clf_perceptron.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

pred_GaussianNB = clf_GaussianNB.predict(X)
acc_GaussianNB = accuracy_score(Y, pred_GaussianNB) * 100
print('Accuracy for GaussianNB: {}'.format(acc_GaussianNB))

index = np.argmax([acc_tree, acc_svc, acc_per, acc_KNN,acc_GaussianNB])
classifiers = {0: 'Decision Tree', 1: 'SVM', 2: 'Perceptron', 3: 'KNN', 4: 'GaussianNB'}
print('Best accuracy is given by {}'.format(classifiers[index]))

print(clf_tree.predict([[175, 65, 40]]))
