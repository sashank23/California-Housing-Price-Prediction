import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load the dataset
data = pd.read_csv("housing.csv")
data.dropna(inplace=True)
data.info()

# Split the data into features (X) and target (y)
X = data.drop(["median_house_value"], axis=1)
y = data['median_house_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_data = X_train.join(y_train)

# Visualize the distributions of the features
train_data.hist(figsize=(15, 8))
plt.show()

# Log transform skewed features
train_data["total_rooms"] = np.log(train_data["total_rooms"] + 1)
train_data["total_bedrooms"] = np.log(train_data["total_bedrooms"] + 1)
train_data["population"] = np.log(train_data["population"] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

# Visualize the transformed distributions of the features
train_data.hist(figsize=(15, 8))
plt.show()

# Convert categorical variable 'ocean_proximity' into dummy variables
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity).astype(int)).drop(['ocean_proximity'], axis=1)

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(15, 8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
plt.show()

# Plot scatterplot of latitude vs longitude, colored by median_house_value
plt.figure(figsize=(15, 8))
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")
plt.show()

# Prepare training data
X_train = train_data.drop(["median_house_value"], axis=1)
y_train = train_data["median_house_value"]

# Train a linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Prepare testing data
test_data = X_test.join(y_test)
test_data["total_rooms"] = np.log(test_data["total_rooms"] + 1)
test_data["total_bedrooms"] = np.log(test_data["total_bedrooms"] + 1)
test_data["population"] = np.log(test_data["population"] + 1)
test_data["households"] = np.log(test_data["households"] + 1)
test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity).astype(int)).drop(["ocean_proximity"], axis=1)
X_test, y_test = test_data.drop(["median_house_value"], axis=1), test_data["median_house_value"]

# Evaluate the linear regression model
print("Linear Regression score:", reg.score(X_test, y_test))

# Train a random forest regressor
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
print("Random Forest score:", forest.score(X_test, y_test))

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "n_estimators": [3, 10, 30, 25],
    "max_features": [2, 4, 6, 8],
}
grid_search = GridSearchCV(forest, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(X_train, y_train)

# Evaluate the best random forest model
best_forest = grid_search.best_estimator_
print("Best Random Forest score:", best_forest.score(X_test, y_test))
