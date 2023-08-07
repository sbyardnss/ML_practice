# going to use the Boston Housing Dataset
# was removed due to racial insensitivity. 
# pulling raw dataset from github for following tutorial. 
# https://github.com/selva86/datasets/blob/master/BostonHousing.csv
# https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
# http://lib.stat.cmu.edu/datasets/boston

from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd

url = "https://github.com/selva86/datasets/blob/master/BostonHousing.csv?raw=true"
data = pd.read_csv(url, index_col=0)
print(data.head(1))
# california = fetch_california_housing()
# X = california.data
# y = california.target
# feature_names = [
#     "MedInc",
#     "HouseAge",
#     "AveRooms",
#     "AveBedrms",
#     "Population",
#     "AveOccup",
#     "Latitude",
#     "Longitude",
# ]
# print(california)
