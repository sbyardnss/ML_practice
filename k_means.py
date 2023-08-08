# SIGMOID algorithm
# useful in learning non-linear relationships between data points
# a sigmoid function is a regression function used to find
# correlation in data patterns that aren't exactly linear
# often the graph of a sigmoid looks like an S curve
# sigmoid responses always between 0 and 1 (?) so
# useful when looking for probability

# Kmeans clustering: math explanation in video. look up reasoning elsewhere
#   - uses centroids created at random points.
#       - then creates hyperplane between the two and
#         gets average position of data points on either side of said plane
#       - this process is repeated until there is a plane created between the clusters
#   -
# clustering: the task of separating unlabeled data points in to different
# clusters such that similar data points fall in the same cluster than those
# which differ from the others
#   - difference between classification and clustering is that classification
#     models are trained with labels and clustering models are not

# this example is a KMeans clustering algorithm
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.preprocessing import scale
import pandas as pd

# all data brought in
bc = load_breast_cancer()
# data for inputs
X = scale(bc.data)
# target output for X
y = bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# random state adjusts randomization? research more later
model = KMeans(n_clusters=2, random_state=0)
# no y_train here. clustering does not use labels
# the model will instead separate them into clusters on its own
model.fit(X_train)
predictions = model.predict(X_test)
labels = model.labels_
# print("predictions: ", predictions)
# print("labels: ", labels)
# print('accuracy: ', metrics.accuracy_score(y_test, predictions))
# print("actual: ", y_test)
# with Kmeans not using labels, there is about an even chance that the accuracy
# will be inverted.

# to fix this issue we will use a cross tabulation test

# not sure how to read this correctly. seems roughly the same
# regardless of whether issue was present or not
# print(pd.crosstab(y_train, labels))

# another way to calculate accuracy
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
# not sure how this works. need to research
bench_k_means(model, 'k_means', X)