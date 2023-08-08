# Boston housing dataset used in tutorial removed from sklearn
# due to racial insensitivity.
# using ames and california housing market data instead

from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# ames housing data. many more features so maybe a svm for examining?
ames = pd.read_csv('ames_house_data/train.csv', index_col=[0])
ames_test = pd.read_csv('ames_house_data/test.csv', index_col=[0])

ames_features = ames.columns
# print(ames_features)
ames_target = ames['SalePrice'].values
ames_useful = ames[[
    'OverallQual',
    'YearBuilt',
    'YearRemodAdd',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'GrLivArea',
    'FullBath',
    'TotRmsAbvGrd',
    'GarageCars',
    'GarageArea'
]].values

california = fetch_california_housing()

X = california.data
y = california.target
df = pd.DataFrame(X)
# OverallQual, YearBuilt, YearRemodAdd, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GrLivArea,
# FullBath, TotRmsAbvGrd, GarageYrBlt, GarageCars, GarageArea
# ames features
# ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
#        'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
#        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
#        'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
#        'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
#        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
#        'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
#        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
#        'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
#        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
#        'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
#        'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
#        'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
#        'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
#        'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
#        'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice'],
#       dtype='object')
# features of california
# - MedInc        median income in block group
# - HouseAge      median house age in block group
# - AveRooms      average number of rooms per household
# - AveBedrms     average number of bedrooms per household
# - Population    block group population
# - AveOccup      average number of household members
# - Latitude      block group latitude
# - Longitude     block group longitude
cali_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude']
cali_target_names = ['MedHouseVal']

# algorithm
l_reg = linear_model.LinearRegression()
# print(ames['Alley'].values)
# plt.scatter((X.T[0]), y)
# plt.show()

# train test split for ames data
ames_useful_X_train, ames_useful_X_test, ames_useful_y_train, ames_useful_y_test = train_test_split(ames_useful, ames_target, test_size=0.2 )

ames_model = l_reg.fit(ames_useful_X_train, ames_useful_y_train)
ames_predictions = ames_model.predict(ames_useful_X_test)
# print("ames_predictions:", ames_predictions)
print("R^2 value: ", l_reg.score(ames_useful, ames_target)) #77% accuracy
# train test split for cali data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)
# show data for calij
# print("predictions:", predictions)
# print("R^2 value:", l_reg.score(X, y)) #60% accuracy
# print("coedd:", l_reg.coef_)
# print("intercept: ", l_reg.intercept_)