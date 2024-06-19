import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# to avoid warnings
import warnings
warnings.filterwarnings('ignore')
# Shape of Data
house.shape
# Data information
house.info()
# Checking Null Values
house.isna().sum()
# Checking Duplicate Values
house.duplicated().sum()
# Summary of data
house.describe()
house.columns
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of Housing Prices (MEDV)

sns.distplot(house['MEDV'], bins=20, kde=True)
plt.title('Distribution of Housing Prices (MEDV)')
plt.xlabel('Median Housing Price ($1000s)')
plt.ylabel('Frequency')
plt.show()
# List of features
features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']

# Create a scatter plot and boxplot for each feature side by side
for feature in features:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  # Create a new figure with 1 row and 2 columns

    # Scatter plot of feature with target variable
    axes[0].scatter(house[feature], house['MEDV'])
    axes[0].set_title(f'Scatter plot of {feature} with House Price')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('House Price')

    # Boxplot of the feature
    axes[1].boxplot(house[feature])
    axes[1].set_title(f'Boxplot for {feature}')
    axes[1].set_xlabel('Feature')
    axes[1].set_ylabel('Values')

    plt.tight_layout()
    plt.show();
    rad_medv_mean = house.groupby('RAD')['MEDV'].mean().reset_index()
rad_medv_mean
# Bar Plot Average House Price By Accessibility of Road Highways

sns.barplot(x='RAD', y='MEDV', data=rad_medv_mean, color='orange',edgecolor='black')
plt.title('Average House Price By Accessibility of Road Highways')
plt.xlabel('Accessibility of Road Highways')
plt.ylabel('Mean Housing Price ($1000s)')
plt.show();
# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(house.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Boston Housing Features')
plt.show()
# Split the Data
X = house.drop(columns=['MEDV']) #features
y = house['MEDV'] #target variable
# Splitting Data for Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=2)
# shape of spiltted data
print("The shape of X_train :",X_train.shape)
print("The shape ofX_test :",X_test.shape)
print("The shape of y_train :",y_train.shape)
print("The shape of y_test :",y_test.shape)
# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=5)
# Fit the model on Training dataset
dtr.fit(X_train,y_train)
# Predictions of  decision Tree Regressor on Testing Data
y_pred_dtr=dtr.predict(X_test)
# Accuracy Score of Model
from sklearn.metrics import mean_absolute_percentage_error
error = mean_absolute_percentage_error(y_pred_dtr,y_test)
print("Accuracy of Decision Tree Regressor is :%.2f "%((1 - error)*100),'%')
# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth = 10, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200)
# Fit the model on Training datset
rfr.fit(X_train,y_train)
# Predictions of  Ranforest Forest Regressor on Testing Data
y_pred_rfr = rfr.predict(X_test)
# Accuracy Score of Model

error = mean_absolute_percentage_error(y_pred_rfr,y_test)
print("Accuracy of Random Forest Regressor is :%.2f "%((1 - error)*100),'%')
