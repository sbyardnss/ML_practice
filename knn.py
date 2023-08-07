import numpy as np
import pandas as pd
from sklearn import metrics, neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# this will be a classification model using the KNN algorithm for learning
# knn: k-nearest neighbors algorithm. a non-parametric, supervised learning
# classifier, which uses proximity to make classifications or predictions
# about the grouping of an individual data point

# create table from data in car.data
# added first line in car.data to act as table head

data = pd.read_csv('car.data')
# car dataset url
# https://archive.ics.uci.edu/dataset/19/car+evaluation


# print(data.head())

# select properties (features) to examine for learning
X = data[[
    'buying',
    'maintenance',
    'safety'
]].values
# select column that corresponds to evaluation of features
# the class column tell us the condition of the car as a whole. 
# based on this, our model can examine the features of each and
# learn what feature values correspond to an acceptable car
y= data[['class']]

# print(X, y)

# need to convert data in X into numeric values using LabelEncoder()
# (currently given values such as 'low' and 'high' etc)
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])
# print(X)

# convert data in y using mapping
# we are declaring exactly what the value of each will be 
# based on the values currently in y.
# the converted data in X doesn't need to be stricly definied 
# because the model will simply recognize the class result in y 
# from particular data values in X
# example: 'high' in X is 0 and 'low' is 1. this doesn't make sense 
# with a middle ground between them but the model simply needs to recognize 
# where similar data values are in future features. as in it recognizes that 
# if there are all 0 in the data columns then that corresponds with the 
# class result from a car with all 'high' values

label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}
y['class'] = y['class'].map(label_mapping)
y=np.array(y)
# print(y)

# now that data is in proper format, we can begin creating the model

# check url below for info on how this is working
# https://scikit-learn.org/stable/modules/neighbors.html
# creating a knn object for the algorithm
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

# not yet ready. we need to train the model
# training needs an X parameter (features) and y parameter (labels)
# to ensure the accuracy of our model, we are going to 
# separate our data into 'training' data and 'testing' data

# this creates our training and testing data. the test_size means 20% of our data will be used to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# this trains the data i believe?
knn.fit(X_train, y_train)
# this version removes an issue with the data prediction on one value. Not sure why
# knn.fit(X_train, y_train.ravel())


# making predictions from training data using the testing data
prediction = knn.predict(X_test)

# now see how close our predictions match the correct results
accuracy = metrics.accuracy_score(y_test, prediction)
# print("predictions", prediction)
# print("accuracy", accuracy)
# this has created a machine learning model with roughly 75% accuracy

a = 1727

# here is a single actual value from the class data at index a
# print("actual value", y[a])

# here is the prediction from our model when provided the corresponding features
# for y at index a

# print("predicted value", knn.predict(X)[a])





