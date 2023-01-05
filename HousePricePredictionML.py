import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

# To display more than 5 columns of data, in run terminal of pycharm
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 12)

df = pd.read_csv("train.csv")  # reading the imported csv file
# EDA- Exploratory Data Analysis
# df.head() - Looking at the 1st 5 rows of the data set
# df.shape - Number of rows and columns (size of data)
# print(df.info()) - The number of non null values and dtypes
# print(df.describe().T) - Getting the statistical summary of the dataset

# Visualising correaltions between variables
# plt.figure(figsize=(12,10))
# sns.heatmap(df.corr(), cmap="BuPu")
# plt.title("Correation between variable", size= 8)
# plt.show()
# print(df.corr()["YearBuilt"]) - To display a specific columns in a table

