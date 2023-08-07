# Boston housing dataset used in tutorial removed from sklearn 
# due to racial insensitivity. 
# using california housing market data instead

from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd


california = fetch_california_housing()
X = california.data
y = california.target
# features of california
# - MedInc        median income in block group
# - HouseAge      median house age in block group
# - AveRooms      average number of rooms per household
# - AveBedrms     average number of bedrooms per household
# - Population    block group population
# - AveOccup      average number of household members
# - Latitude      block group latitude
# - Longitude     block group longitude
print(y)