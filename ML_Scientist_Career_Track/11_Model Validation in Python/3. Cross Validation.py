import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 8)

# The problems with holdout sets
'''
Two samples

After building several classification models based on the tic_tac_toe dataset, you realize that some models do not 
generalize as well as others. You have created training and testing splits just as you have been taught, so you are curious why your validation process is not working.

After trying a different training, test split, you noticed differing accuracies for your machine learning model. 
Before getting too frustrated with the varying results, you have decided to see what else could be going on.
'''
tic_tac_toe = pd.read_csv('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                          '11_Model Validation in Python/data/tic-tac-toe.csv')
print(tic_tac_toe.head())


# Create two different samples of 200 observations
sample1 = tic_tac_toe.sample(n=200, random_state=1111)
sample2 = tic_tac_toe.sample(n=200, random_state=1171)

# Print the number of common observations
print(len([index for index in sample1.index if index in sample2.index]))

# Print the number of observations in the Class column for both samples
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts())


# Cross-validation
'''
scikit-learn's KFold()

You just finished running a colleagues code that creates a random forest model and calculates an out-of-sample accuracy. You noticed that your colleague's code did not have a random state, and the errors you found were completely different than the errors your colleague reported.

To get a better estimate for how accurate this random forest model will be on new data, you have decided to generate some indices to use for KFold cross-validation.
'''
candy = pd.read_csv('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                    '11_Model Validation in Python/data/candy-data.csv')
print(candy.head())

# X, y = ndarray
X = candy.drop(['competitorname', 'winpercent'], axis=1).to_numpy()
y = candy['winpercent'].to_numpy()

from sklearn.model_selection import KFold

# Use KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1111)

# Create splits
splits = kf.split(X)

# Print the number of indices
for train_index, val_index in splits:
    print("Number of training indices: %s" % len(train_index))
    print("Number of validation indices: %s" % len(val_index))


# Using KFold indices
'''
You have already created splits, which contains indices for the candy-data dataset to complete 5-fold cross-validation. 
To get a better estimate for how well a colleague's random forest model will perform on a new data, 
you want to run this model on the five different training and validation indices you just created.

In this exercise, you will use these indices to check the accuracy of this model using the five different splits. A for loop has been provided to assist with this process.
'''
# Create splits
splits = kf.split(X)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)

# Access the training and validation indices of splits
for train_index, val_index in splits:
    # Setup the training and validation data
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]

    # Fit the random forest model
    rfc.fit(X_train, y_train)

    # Make predictions, and print the accuracy
    predictions = rfc.predict(X_val)
    print("Split accuracy: " + str(mean_squared_error(y_val, predictions)))


# sklearn's cross_val_score()
'''
scikit-learn's methods

You have decided to build a regression model to predict the number of new employees your company will successfully hire next month. You open up a new Python script to get started, but you quickly realize that sklearn has a lot of different modules. Let's make sure you understand the names of the modules, the methods, and which module contains which method.
Follow the instructions below to load in all of the necessary methods for completing cross-validation using sklearn. You will use modules:

- metrics
- model_selection
- ensemble
'''
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer

# Implement cross_val_score()
'''
Your company has created several new candies to sell, but they are not sure if they should release all five of them. 
To predict the popularity of these new candies, you have been asked to build a regression model u
sing the candy dataset. Remember that the response value is a head-to-head win-percentage against other candies.

Before you begin trying different regression models, you have decided to run cross-validation on a simple 
random forest model to get a baseline error to compare with any future results.
'''
rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
mse = make_scorer(mean_squared_error)

# Setup cross_val_score
cv = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10, scoring=mse)

# Print the mean error
print(cv.mean())


# Leave-one-out-cross-validation (LOOCV)
'''
When to use LOOCV?
    The amount of training data is limited
    You want the absolute best error estimate for new data

Be cautious when:
    Computation resources are limited
    You have a lot of data
    You have a lot of parameters to test
'''

# Leave-one-out-cross-validation
from sklearn.metrics import mean_absolute_error
# Create scorer
mae_scorer = make_scorer(mean_absolute_error)

rfr = RandomForestRegressor(n_estimators=15, random_state=1111)

# Implement LOOCV
scores = cross_val_score(rfr, X, y, cv=85, scoring=mae_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))

