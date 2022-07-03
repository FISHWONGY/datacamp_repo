import pandas as pd
import numpy as np

# Introduction to hyperparameter tuning
'''
Model Parameters
    Learned or estimated from the data
    The result of fitting a model
    Used when making future predictions
    Not manually set
Model Hyperparameters
    Manually set before the training occurs
    Specify how the training is supposed to happen
Hyperparameter tuning
    Select hyperparameters
    Run a single model type at different value sets
    Create ranges of possible values to select from
    Specify a single accuracy metric
'''

# Creating Hyperparameters
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators='warn', max_features='auto', random_state=1111)
print(rfr.get_params())


# Maximum Depth
max_depth = [4, 8, 12]

# Minimum samples for a split
min_samples_split = [2, 5, 10]

# Max features
max_features = [4, 6, 8, 10]


'''
Running a model using ranges

You have just finished creating a list of hyperparameters and ranges to use when tuning a predictive model for an assignment.
'''
import random

# Fill in rfr using your variables
rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=random.sample(max_depth, 1)[0],
    min_samples_split=random.sample(min_samples_split, 1)[0],
    max_features=random.sample(max_features, 1)[0]
)

# Print out the parameters
print(rfr.get_params())


'''
RandomizedSearchCV
Grid Search
    Benefits
        Tests every possible combination
    Drawbacks
        Additional hyperparameters increase training time exponentially
Alternatives
    Random searching
    Bayesian optimization
'''

# Preparing for RandomizedSearch
from sklearn.metrics import make_scorer, mean_squared_error

# Finish the dictionary by adding the max_depth parameter
param_dist = {
    "max_depth": [2, 4, 6, 8],
    "max_features": [2, 4, 6, 8, 10],
    "min_samples_split": [2, 4, 8, 16]
}

# Create a random forest regression model
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)

# Create a scorer to use (use the mean squared error)
scorer = make_scorer(mean_squared_error)


# Implementing RandomizedSearchCV
'''
You are hoping that using a random search algorithm will help you improve predictions for a class assignment. You professor has challenged your class to predict the overall final exam average score.

In preparation for completing a random search, you have created:
    param_dist: the hyperparameter distributions
    rfr: a random forest regression model
    scorer: a scoring method to use
'''
from sklearn.model_selection import RandomizedSearchCV

# Build a random search using param_dist, rfr, and scorer
random_search = RandomizedSearchCV(estimator=rfr,
                                   param_distributions=param_dist,
                                   n_iter=10,
                                   cv=5,
                                   scoring=scorer
                                   )


# Selecting your final model
'''
Best classification accuracy

You are in a competition at work to build the best model for predicting the winner of a Tic-Tac-Toe game. 
You already ran a random search and saved the results of the most accurate model to rs.
'''
tic_tac_toe = pd.read_csv('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                          '11_Model Validation in Python/data/tic-tac-toe.csv')
# Create dummy variables using pandas
X = pd.get_dummies(tic_tac_toe.iloc[:, 0:9])
y = tic_tac_toe.iloc[:, 9]
y = tic_tac_toe['Class'].apply(lambda x: 1 if x == 'positive' else 0)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_features='auto')

param_dist = {
    'max_depth': [2, 4, 8, 12],
    'min_samples_split': [2, 4, 6, 8],
    'n_estimators': [10, 20, 30]
}

rs = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist, n_iter=10,
                        cv=5, scoring=None, random_state=1111)

rs.fit(X, y)
print(rs.best_params_)


# Selecting the best precision model
'''
Your boss has offered to pay for you to see three sports games this year. Of the 41 home games your favorite team plays, you want to ensure you go to three home games that they will definitely win. You build a model to decide which games your team will win.

To do this, you will build a random search algorithm and focus on model precision (to ensure your team wins). You also want to keep track of your best model and best parameters, so that you can use them again next year (if the model does well, of course).
'''
sports = pd.read_csv('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                     '11_Model Validation in Python/data/sports.csv')
X = sports.drop('win', axis=1)
y = sports['win']

rfc = RandomForestClassifier()

param_dist = {
    'max_depth': range(2, 12, 2),
    'min_samples_split': range(2, 12, 2),
    'n_estimators': [10, 25, 50]
}

from sklearn.metrics import precision_score

# Create a precision scorer
precision = make_scorer(precision_score)

# Finalize the random search
rs = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist,
                        scoring=precision, cv=5, n_iter=10,
                        random_state=1111)

rs.fit(X, y)

# Print the mean test scores:
print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# Print the best model scores:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))
# To get the best params
print(rs.best_params_)

# Your model's precision was 91%! The best model accurately predicts a winning game 91% of the time.
# If you look at the mean test scores, you can tell some of the other parameter sets did really poorly.
# Also, since you used cross-validation, you can be confident in your predictions. Well done!
