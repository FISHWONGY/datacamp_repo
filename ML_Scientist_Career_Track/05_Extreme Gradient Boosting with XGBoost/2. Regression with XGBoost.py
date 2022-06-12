import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

plt.rcParams['figure.figsize'] = (7, 7)

# Regression review
'''
Common regression metrics
 - Root Mean Squared Error (RMSE)
 - Mean Absolute Erro (MAE)
'''
# Objective (loss) functions and base learners
'''
Objective functions and Why we use them
 - Quantifies how far off a prediction is from the actual result
 - Measures the difference between estimated and true values for some collection of data
 - Goal: Find the model that yields the minimum value of the loss function

Common loss functions and XGBoost
 - Loss function names in xgboost:
   - reg:linear - use for regression problems
   - reg:logistic - use for classification problems when you want just decision, not probability
   - binary:logistic - use when you want probability rather than just decision

Base learners and why we need them
 - XGBoost involves creating a meta-model that is composed of many individual models that combine to give a final prediction
 - Individual models = base learners
 - Want base learners that when combined create final prediction that is non-linear
 - Each base learner should be good at distinguishing or predicting different parts of the dataset
 - Two kinds of base learners: tree and linear
'''

# Decision trees as base learners
df = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                 '05_Extreme Gradient Boosting with XGBoost/data/ames_housing_trimmed_processed.csv')
X, y = df.iloc[:, :-1], df.iloc[:, -1]

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiatethe XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', seed=123, n_estimators=10)

# Fit the regressor to the training set
xg_reg.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)

# compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


###
# Linear base learners
'''
because it's uncommon, you have to use XGBoost's own non-scikit-learn compatible functions 
to build the model, such as xgb.train().

In order to do this you must create the parameter dictionary that describes the kind of 
booster you want to use (similarly to how you created the dictionary in Chapter 1 when you used xgb.cv()). 
The key-value pair that defines the booster type (base model) you need is "booster":"gblinear".

Once you've created the model, you can use the .train() and .predict() methods of the model 
just like you've done in the past.
'''
# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)

# Create the parameter dictionary: params
params = {"booster":"gblinear", "objective":"reg:squarederror"}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=5)

# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# Evaluating model quality
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Perform cross-valdiation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4,
                    num_boost_round=5, metrics='rmse', as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results['test-rmse-mean']).tail(1))


# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Perform cross-valdiation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4,
                    num_boost_round=5, metrics='mae', as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results['test-mae-mean']).tail(1))

# Regularization and base learners in XGBoost
'''
Regularization in XGBoost
 - Regularization is a control on model complexity
 - Want models that are both accurate and as simple as possible
 - Regularization parameters in XGBoost:
    - Gamma - minimum loss reduction allowed for a split to occur
    - alpha - L1 regularization on leaf weights, larger values mean more regularization
    - lambda - L2 regularization on leaf weights

Base learners in XGBoost
 - Linear Base learner
    - Sum of linear terms
    - Boosted model is weighted sum of linear models (thus is itself linear)
    - Rarely used
 - Tree Base learner
    - Decision tree
    - Boosted model is weighted sum of decision trees (nonlinear)
    - Almost exclusively used in XGBoost
'''

# Using regularization in XGBoost
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

reg_params = [1, 10, 100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective": "reg:squarederror", "max_depth": 3}

# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

# Iterate over reg_params
for reg in reg_params:
    # Update l2 strength
    params['lambda'] = reg

    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                             num_boost_round=5, metrics='rmse', as_pandas=True, seed=123)

    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse['test-rmse-mean'].tail(1).values[0])

# Loot at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))


# Visualizing individual XGBoost trees
# XGBoost has a plot_tree() function that makes this type of visualization easy.
# Once you train a model using the XGBoost learning API, you can pass it to the plot_tree() function
# along with the number of trees you want to plot using the num_trees argument.
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameters dictionary: params
params = {"objective":'reg:squarederror', 'max_depth':2}

# Train the model: xg_reg
xg_reg = xgb.train(dtrain=housing_dmatrix, params=params, num_boost_round=10)

# Plot the first tree
fig, ax = plt.subplots(figsize=(15, 15))
xgb.plot_tree(xg_reg, num_trees=0, ax=ax);

# Plot the fifth tree
fig, ax = plt.subplots(figsize=(15, 15))
xgb.plot_tree(xg_reg, num_trees=4, ax=ax);

# Plot the last tree sideways
fig, ax = plt.subplots(figsize=(15, 15))
xgb.plot_tree(xg_reg, rankdir="LR", num_trees=9, ax=ax)


###
# Visualizing feature importances: What features are most important in my dataset
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Train the model: xg_reg
xg_reg = xgb.train(dtrain=housing_dmatrix, params=params, num_boost_round=10)

# Plot the feature importance
xgb.plot_importance(xg_reg);