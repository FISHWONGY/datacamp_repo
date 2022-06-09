import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Tuning a CART's Hyperparameters
'''
Hyperparameters
 - Machine learning model:
    - parameters: learned from data
      - CART example: split-point of a node, split-feature of a node, ...
    - hyperparameters: not learned from data, set prior to training
      - CART example: max_depth, min_samples_leaf, splitting criterion, ...

What is hyperparameter tuning?
 - Problem: search for a set of optimal hyperparameters for a learning algorithm.
 - Solution: find a set of optimal hyperparameters that results in an optimal model.
 - Optimal model: yields an optimal score
 - Score : defaults to accuracy (classification) and  (regression)
 - Cross-validation is used to estimate the generalization performance.

Approaches to hyperparameter tuning
 - Grid Search
 - Random Search
 - Bayesian Optimization
 - Genetic Algorithm

Grid search cross validation
 - Manually set a grid of discrete hyperparameter values.
 - Set a metric for scoring model performance.
 - Search exhaustively through the grid.
 - For each set of hyperparameters, evaluate each model's CV score
 - The optimal hyperparameters are those of the model achieving the best CV score.
'''

# Tree hyperparameters
indian = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                     '04_Machine Learning with Tree-Based Models in Python/data/'
                     'indian_liver_patient_preprocessed.csv', index_col=0)

X = indian.drop('Liver_disease', axis='columns')
y = indian['Liver_disease']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.tree import DecisionTreeClassifier

# Instantiate dt
dt = DecisionTreeClassifier()

# Check default hyperparameter
dt.get_params()


# Set the tree's hyperparameter grid
# Define params_dt
params_dt = {
    'max_depth': [2, 3, 4],
    'min_samples_leaf': [0.12, 0.14, 0.16, 0.18],
}

# Search for the optimal tree
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='roc_auc', cv=5, n_jobs=-1)

grid_dt.fit(X_train, y_train)


# Evaluate the optimal tree
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Extract the best hyperparameters
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters:\n', best_hyperparams)
# Extract best CV score
best_cv_score = grid_dt.best_score_
print('Best CV accuracy:\n', best_cv_score)
# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print("Test set ROC AUC score: {:.3f}".format(test_roc_auc))


###
# Tuning a RF's Hyperparameters
'''
Random Forest Hyperparameters
 - CART hyperparameters
 - number of estimators
 - Whether it uses bootstrapping or not

Tuning is expensive
 - Hyperparameter tuning:
    - Computationally expensive,
    - sometimes leads to very slight improvement
 - Weight the impact of tuning on the whole project
'''

# Random forests hyperparameters
bike = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                   '04_Machine Learning with Tree-Based Models in Python/data/bikes.csv')

X = bike.drop('cnt', axis='columns')
y = bike['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor()

# Get hyperparameters
rf.get_params()


# Set the hyperparameter grid of RF
# Define the dicrionary 'params_rf'
params_rf = {
    'n_estimators': [100, 350, 500],
    'max_features': ['log2', 'auto', 'sqrt'],
    'min_samples_leaf': [2, 10, 30],
}


# Search for the optimal forest
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error', cv=3,
                       verbose=1, n_jobs=-1)

# fit model
grid_rf.fit(X_train, y_train)


# Evaluate the optimal forest
from sklearn.metrics import mean_squared_error as MSE

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred) ** 0.5

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test))

