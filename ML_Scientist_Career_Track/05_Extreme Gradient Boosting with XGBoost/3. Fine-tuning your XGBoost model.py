import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

plt.rcParams['figure.figsize'] = (10, 10)

# Tuning the number of boosting rounds
df = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                 '05_Extreme Gradient Boosting with XGBoost/data/ames_housing_trimmed_processed.csv')
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Creata the parameter dictionary for each tree: params
params = {"objective": "reg:squarederror",
          "max_depth": 3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Interate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3,
                        num_boost_round=curr_num_rounds, metrics='rmse',
                        as_pandas=True, seed=123)

    # Append final round RMSE
    final_rmse_per_round.append(cv_results['test-rmse-mean'].tail().values[-1])

# Print the result DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses, columns=['num_boosting_rounds', 'rmse']))


# Automated boosting round selection using early_stopping
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective": "reg:squarederror", "max_depth": 4}

# Perform cross-validation with early-stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, nfold=3, params=params, metrics="rmse",
                    early_stopping_rounds=10, num_boost_round=50, as_pandas=True, seed=123)

# Print cv_results
print(cv_results)


###
# Overview of XGBoost's hyperparameters
'''
Common tree tunable parameters
 - learning rate: learning rate/eta
 - gamma: min loss reduction to create new tree split
 - lambda: L2 regularization on leaf weights
 - alpha: L1 regularization on leaf weights
 - max_depth: max depth per tree
 - subsample: % samples used per tree
 - colsample_bytree: % features used per tree

Linear tunable parameters
 - lambda: L2 reg on weights
 - alpha: L1 reg on weights
 - lambda_bias: L2 reg term on bias

You can also tune the number of estimators used for both base model types!
'''

# Tuning eta
# The learning rate in XGBoost is a parameter that can range between 0 and 1,
# with higher values of "eta" penalizing feature weights more strongly, causing much stronger regularization.

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective": "reg:squarederror", "max_depth": 3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematicallyvary the eta
for curr_val in eta_vals:
    params['eta'] = curr_val

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3,
                        early_stopping_rounds=5, num_boost_round=10, metrics='rmse', seed=123,
                        as_pandas=True)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results['test-rmse-mean'].tail().values[-1])

# Print the result DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=['eta', 'best_rmse']))
# 0.010 is the best eta


# Tuning max_depth
# tune max_depth, which is the parameter that dictates the maximum depth that each tree in a boosting round can grow to.
# Smaller values will lead to shallower trees, and larger values to deeper trees
# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary
params = {"objective": "reg:squarederror"}

# Create list of max_depth values
max_depths = [2, 5, 10, 20]
best_rmse = []

for curr_val in max_depths:
    params['max_depth'] = curr_val

    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                        early_stopping_rounds=5, num_boost_round=10, metrics='rmse', seed=123,
                        as_pandas=True)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results['test-rmse-mean'].tail().values[-1])

# Print the result DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)), columns=['max_depth', 'best_rmse']))
# 5 is the best max depth

# Tuning colsample_bytree
# In xgboost, colsample_bytree must be specified as a float between 0 and 1.
# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary
params = {"objective": "reg:squarederror", "max_depth": 3}

# Create list of hyperparameter values: colsample_bytree_vals
colsample_bytree_vals = [0.1, 0.5, 0.8, 1]
best_rmse = []

# Systematically vary the hyperparameter value
for curr_val in colsample_bytree_vals:
    params['colsample_bytree'] = curr_val

    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                        num_boost_round=10, early_stopping_rounds=5,
                        metrics="rmse", as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)),
                   columns=["colsample_bytree", "best_rmse"]))
# 1 is the best colsample_bytree

###
# Review of grid search and random search
'''
Grid search: review
 - Search exhaustively over a given set of hyperparameters, once per set of hyperparameters
 - Number of models = number of distinct values per hyperparameter multiplied across each hyperparameter
 - Pick final model hyperparameter values that give best cross-validated evaluation metric value

Random search: review
 - Create a (possibly infinte) range of hyperparameter values per hyperparameter that you would like to search over
 - Set the number of iterations you would like for the random search to continue
 - During each iteration, randomly draw a value in the range of specified values for each hyperparameter 
   searched over and train/evaluate a model with those hyperparameters
 - After you've reached the maximum number of iterations, select the hyperparameter configuration 
   with the best evaluated score
'''

# Grid search with XGBoost
from sklearn.model_selection import GridSearchCV

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(param_grid=gbm_param_grid, estimator=gbm,
                        scoring='neg_mean_squared_error', cv=4, verbose=1)

# Fit grid_mse to the data
grid_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
# RESULT
# Best parameters found:  {'colsample_bytree': 0.7, 'max_depth': 2, 'n_estimators': 50}
# Lowest RMSE found:  30355.698207097197


# Random search with XGBoost
# Often, GridSearchCV can be really time consuming, so in practice, you may want to use RandomizedSearchCV instead
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators=10)

# Perform random search: randomized_mse
randomized_mse = RandomizedSearchCV(param_distributions=gbm_param_grid, estimator=gbm,
                                    scoring='neg_mean_squared_error', n_iter=5, cv=4,
                                    verbose=1)

# Fit randomized_mse to the data
randomized_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
# RESULT
# Best parameters found:  {'n_estimators': 25, 'max_depth': 4}
# Lowest RMSE found:  29998.4522530019

# Limits of grid search and random search
'''
limitations
 - Grid Search
    - Number of models you must build with every additionary new parameter grows very quickly
 - Random Search
    - Parameter space to explore can be massive
    - Randomly jumping throughtout the space looking for a "best" results becomes a waiting game
'''