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

# Visualising correlations between variables
# plt.figure(figsize=(12,10))
# sns.heatmap(df.corr(), cmap="BuPu")
# plt.title("Correation between variable", size= 8)
# plt.show()
# print(df.corr()["YearBuilt"]) - To display a specific columns in a table

# We are selecting numerical features which have more than 0.50 or less than -0.50 correlation rate based on Pearson
# Correlation Method—which is the default value of parameter "method" in corr() function. As for selecting
# categorical features, I selected the categorical values which I believe have significant effect on the target
# variable such as Heating and MSZoning.
imp_num_colns = list(df.corr()["SalePrice"][(df.corr()["SalePrice"] > 0.50) | (df.corr()["SalePrice"] < -0.50)].index)
cat_cols = ["MSZoning", "Utilities", "BldgType", "Heating", "KitchenQual", "SaleCondition", "LandSlope"]
imp_colmns = imp_num_colns + cat_cols
df = df[imp_colmns]
# print(df)
# print(df.shape)

# print("Missing Values by Columns")
# print("$" * 100)
# print(df.isna().sum())
# print("$" * 100)
# print("Total missing Values: ", df.isna().sum().sum())


# sns.pairplot(df[imp_colmns])
# plt.figure(figsize=(10,8))
# plt.show()

# plt.figure(figsize=(10,8))
# sns.jointplot(x=df["OverallQual"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["YearBuilt"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["YearRemodAdd"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["TotalBsmtSF"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["1stFlrSF"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["GrLivArea"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["FullBath"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["TotRmsAbvGrd"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["GarageCars"], y=df["SalePrice"], kind="kde")
# sns.jointplot(x=df["GarageArea"], y=df["SalePrice"], kind="kde")
# plt.show()

# X, y Split

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# One-Hot Encoding
X = pd.get_dummies(X, columns=cat_cols)

imp_num_colns.remove("SalePrice")
scaler = StandardScaler()
X[imp_num_colns]= scaler.fit_transform(X[imp_num_colns])
print(X)
