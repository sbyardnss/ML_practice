# the goal of neural networks is to find patterns
# neural network
#   - a method in AI that teaches computers to process
#     data in a way that is inspired by the human brain
#   - multiple inputs working towards output
#       - inputs given weight value to change how greatly their value affects output
#           - weight: the importance of an input
#   - output calculated by linear combination
#   - example: 4 inputs (x1-x4) 1 output (y1)
#       - inputs given 'weight' w1-w4 (with respect to name. ie x1 has weight of 1)
#       - x1w1 + x2w2 + x3w3 + x4w4 for summation of inputs
#       - 'activation function' (f(x)) will determine y1 (f(x) = y1)
#   - as there are weights for each input, there are also 'biases' (b1 - b4 respectively)
#       - these biases added to each corresponding value
#       - can factor out biases if necessary
#           - (b1 + b2 + b3 + b4) + X = A (A is what will go into activation
#             function in this example)
#   - this very similar to y = mx + b
#       - b is the bias
#       - m is the weight
#   - 'perceptron' formula for calculating summation of variables with weights and biases
#   - THIS IS ALL A VERY SIMPLE EXPLANATION


#   - for the neural network to recognize patterns, we need to add 'hidden layers'
#       - hidden layers sit between input and output
#       - all inputs connected to all hidden layers

#   - neural networks can be used for regression, clustering, image recognition, etc

# Overfitting and Underfitting
#   - the goal of our model is to be able to predict outcomes from data that the model
#     was not trained on
#   - goal is to have highest accuracy possible
#   - a model that is 'under set' (underfitting) has too much variance
#     and not enough data points to produce consistently accurate predictions
#       - it is too simple
#   - a model that is 'over set' (overfitting) has been too greatly trained according to
#     produce consistently accurate predictions. This can result in a correct
#     prediction being returned as incorrect. the model essentially memorized the
#     training data and made it the hyperplane
#       - it is too complicated


# NEED TO STUDY ALL CONCEPTS BELOW MUCH MUCH FURTHER. CALCULUS MAYBE NECESSARY
# Cost functions and Gradient Descent (study these concepts further)
# - cost function of a neural network tells the neural network how bad it did
#   so that it may train further and improve
#       - mean error is one example (?) (Mean Squared Error)
#           - this is done by taking the mean of the loss functions of each attempt
# - gradient descent - the way that neural networks minimize the cost function
#   after finding it
#   - done by finding the gradient of the cost function
#       - takes in all relevant variables (weights and biases included)
#       - purpose of this gradient is to find which direction the model needs
#       to adjust in order to minimize the cost (either applying greater weight
#       or bias maybe?). either positively or negatively for example


# - Backpropagation - the algorithm used on a training set to calculate
#   the weights or changes needed for certain weights
#   - essentially looking at results to adjust inputs.
#   - normal flow of model is forwardpropagation


# CNN
# Convolutional neural network
#   - convolutional layer (as opposed to normal dense layers) adds a filter
#   that allows it to detect patterns
#       - takes a matrix and finds dot product for each position and compares
#       to neighboring position
#       - this allows the layer to see differences between neighboring elements in
#       multiple ways, clarifying the data a little bit
#       - doing this with more than one matrix allows further comparison
#   - mostly used in image recognition


# example time!
# handwriting digit recognition ai with sklearns neural network
from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist as mnist


# training variables
X_train = mnist.train_images()
y_train = mnist.train_labels()

# testing variables
X_test = mnist.test_images()
y_test = mnist.test_labels()

# print('X_train: ', X_train)
# print('X_test: ', X_test)
# print('y_train: ', y_train)

# check number of dimensions
# print(X_train.ndim)

# 60000 samples of 28x28 pixel images
# print(X_train.shape)

# going to create one list of all pixels
# inputing -1 tells python to use the original value
X_train = X_train.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

# printing this here shows first character for training.
# currently the values range from 0 to 265ish.
# print(X_train[0])

# we are going to change the values to be between 0 and 1
X_train = (X_train/256)
X_test = (X_test/256)
# print(X_train[0])

# study what solver, activation, and hidden layer sizes means 

# DONT RUN THIS. 400% CPU lol
# clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64,64))
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# acc = confusion_matrix(y_test, predictions)
# print(acc)


acc_matrix = [[968, 0, 1, 1, 2, 1, 4, 1, 1, 1],
            [0, 1125, 3, 0, 0, 0, 2, 0, 5, 0],
            [3, 1, 1009, 4, 3, 1, 2, 4, 5, 0],
            [0, 0, 3, 993, 0, 2, 0, 5, 3, 4],
            [1, 0, 6, 0, 961, 0, 2, 3, 0, 9],
            [2, 0, 0, 14, 1, 859, 5, 1, 6, 4],
            [5, 2, 1, 1, 4, 5, 938, 0, 2, 0],
            [0, 4, 10, 3, 2, 0, 0, 1003, 0, 6],
            [5, 1, 3, 7, 4, 2, 4, 2, 942, 4],
            [5, 2, 0, 5, 6, 1, 1, 9, 4, 976]]

def accuracy(cm):
    diagonal = cm.trace()
    elements = cm.sum()
    return diagonal/elements
# print(accuracy(acc))
# 0.9769 accuracy as result