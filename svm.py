# SVM (support vector machine)
#   very effective in high dimensional spaces (many features)
#   many kernel functions
#   used in classification and regression

# svm divides the dataset into classes to 
# find the maximum marginal hyperplane
# dimension of hyperplane depends on number 
# of input features
# the hyperplane is placed in the location where there is 
# the greatest distance between opposing data points (support vectors) from 
# the line location
# as in it looks for the point in the dataset where the largest margin
# can be created and places the hyperplane at the center of that margin

# refer to urls below for better understanding svm with kernel functions
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html#sphx-glr-auto-examples-svm-plot-iris-svc-py
# https://stats.stackexchange.com/questions/90736/the-difference-of-kernels-in-svm

# kernel - function used to increase dimension
# kernels used to work with data even when a 
# straight line hyperplane is not possible
# for example: 
# in dataset with overlapping points from multiple(or all) datasets
# computer will create a third dimension to separate points and
# create a hyperplane in 3d space

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# goal for model is to identify species based on features

iris = datasets.load_iris()
# dataset url
# https://archive.ics.uci.edu/dataset/53/iris

X = iris.data
y = iris.target
# print(X.shape)
# print(y.shape)

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# svc: support vector classifier
model = svm.SVC()
model.fit(X_train, y_train)

# print(model)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print("predictions", predictions)
print("actual:", y_test)
print("accuracy", acc)

# if you want to see the actual names of the flowers
# for i in range(len(predictions)):
#     print("test", i, classes[y_test[i]])
#     print("prediction", i, classes[predictions[i]])

