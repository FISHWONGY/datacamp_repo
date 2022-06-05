import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import seaborn as sns

boston = pd.read_csv('/Online course/datacamp/01_Supervised Learning with scikit-learn/data/boston.csv')

# Creating features and target arrays
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values

# Predicting house value from a single feature
# Get all room data, room is from col 6 of DF, and the 5th value from X, since 0,1,2,3,4,5
X_rooms = X[:, 5]

# Add one more dimension to the array
y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

plt.scatter(X_rooms, y)
plt.ylabel('Value of house / 1000($)')
plt.xlabel('Number of rooms')
plt.show();


# Fitting a regression model
# Build a model
reg = LinearRegression()

# Fit a model
reg.fit(X_rooms, y)

# Check out the regression prediction over the range of the data
# prediction_space = X_new
prediction_space = np.linspace(min(X_rooms),
                               max(X_rooms)).reshape(-1, 1)

# Feed the model to X_new to get prediction (y_new)
prediction = reg.predict(prediction_space)

plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)
plt.show()


# Exercise
# Read the CSV file into a DataFrame: df
df = pd.read_csv('/Online course/datacamp/01_Supervised Learning with scikit-learn/data/gm_2008_region.csv')

# Create arrays for features and target variable
X = df['fertility'].values
y = df['life'].values

# Reshape X and y
y_reshaped = y.reshape(-1, 1)
X_reshaped = X.reshape(-1, 1)


# Linear regression on all features
# Including all the features from the boston housing data set
boston = pd.read_csv('/Online course/datacamp/01_Supervised Learning with scikit-learn/data/boston.csv')

# Creating features and target arrays
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a model
reg_all = LinearRegression()
# fit it to the training set
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
# Get r2 - use .score to the model
reg_all.score(X_test, y_test)

###########################################
# Exercise 1 - Fit & predict for regression
df = pd.read_csv('/Online course/datacamp/01_Supervised Learning with scikit-learn/data/gm_2008_region.csv')

# Create arrays for features and target variable
X_fertility = df['fertility'].values.reshape(-1, 1)
y = df['life'].values.reshape(-1, 1)
# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1, 1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X_fertility, y))

# Plot regression line
sns.scatterplot(x='fertility', y='life', data=df)
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


# Exercise 2 - Train/ test split for regression
X = df['fertility'].values.reshape(-1, 1)
y = df['life'].values.reshape(-1, 1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


################################################
boston = pd.read_csv('/Online course/datacamp/01_Supervised Learning with scikit-learn/data/boston.csv')

# Creating features and target arrays
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values
# Cross-validation
# Cross-validation
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=5)
print(cv_results)
np.mean(cv_results)

# Linear regression minimizes a loss function, chooses a coefficient for each feature variable
# Regularisation - Penalising large coefficient

# Ridge regression -
# Loss function = OLS + alpha*ai^2
# ALpha parameter we need to choose -  similar to picking k for knn, controls model complexity
# when Alpha = 0, get back only OLS (Can lead to overfitting)
# when very high alpha can lead to underfitting

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Normalise = True = making sure all of our variable are on the same scale
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

# Lasso regression
# Loss function = OLS + alpha* |ai|
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)

# Lasso for feature selection
names = boston.drop('MEDV', axis=1).columns
lasso = Lasso(alpha=0.1, normalize=True)
lasso_coef = lasso.fit(X, y).coef_
plt.plot(range(len(names)), lasso_coef)
plt.xticks(range(len(names)), names, rotation=60)
plt.ylabel('Coefficients')
plt.show()

# *** This plot shows the most import feature to our target variable - housing price, is number of rooms (RM)


# EXERCISE 1
df = pd.read_csv('/Online course/datacamp/01_Supervised Learning with scikit-learn/data/gm_2008_region.csv')
# Region is str, needs to change to int instead
region_dict = {
    'Middle East & North Africa': 0,
    'Sub-Saharan Africa': 1,
    'America': 2,
    'Europe & Central Asia': 3,
    'East Asia & Pacific': 4,
    'South Asia': 5
}

df['Region'] = df['Region'].map(region_dict)
# Create arrays for features and target variable
X = df.drop('life', axis='columns').values
y = df['life'].values

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)

df_columns = df.drop('life', axis=1).columns
# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()
# Child mortality is the most import feature to our target variable - life expectancy


# EXERCISE 2
# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []


def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


# Cross-validation (cv) score and std error change over the alpha
# which alpha I should choose?



