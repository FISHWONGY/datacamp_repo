import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Adaboost
'''
Boosting: Ensemble method combining several weak learners to form a strong learner.
 - Weak learner: Model doing slightly better than random guessing
    - E.g., Dicision stump (CART whose maximum depth is 1)
 - Train an ensemble of predictors sequentially.
 - Each predictor tries to correct its predecessor
 - Most popular boosting methods:
    - AdaBoost
    - Gradient Boosting

AdaBoost
 - Stands for Adaptive Boosting
 - Each predictor pays more attention to the instances wrongly predicted by its predecessor.
 - Achieved by changing the weights of training instances.
 - Each predictor is assigned a coefficient  that depends on the predictor's training error
'''

# Define the AdaBoost classifier
indian = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                     '04_Machine Learning with Tree-Based Models in Python/data/'
                     'indian_liver_patient_preprocessed.csv', index_col=0)

X = indian.drop('Liver_disease', axis='columns')
y = indian['Liver_disease']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)


# Train the AdaBoost classifier
# Fit ada to the training set
ada.fit(X_train, y_train)

'''
# Visualizing features importances
# Create a pd.Series of features importances
importances = pd.Series(data=ada.feature_importances_, index=X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
plt.figure(figsize=(12, 8))
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()
'''

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:, 1]


# Evaluate the AdaBoost classifier
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))


###
# Gradient Boosting (GB)
'''
Gradient Boosted Trees
 - Sequential correction of predecessor's errors
 - Does not tweak the weights of training instances
 - Fit each predictor is trained using its predecessor's residual errors as labels
 - Gradient Boosted Trees: a CART is used as a base learner.
'''

# Define the GB regressor
bike = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                   '04_Machine Learning with Tree-Based Models in Python/data/bikes.csv')

X = bike.drop('cnt', axis='columns')
y = bike['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.ensemble import GradientBoostingRegressor

# Instantiate gb
gb = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=2)

# Train the GB regressor
# Fit gb to the training set
gb.fit(X_train, y_train)

# Visualizing features importances
# Create a pd.Series of features importances
importances = pd.Series(data=gb.feature_importances_, index=X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
plt.figure(figsize=(12, 8))
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()
# hr, workingday instant are the 3 most important features


# Parameter - max_depth we need to decide
X = bike[['hr']].values  # sinece HR is the most important features
y = bike['cnt']
# No way to know if it's too deep, we just need to try and plot things out to see
gb = GradientBoostingRegressor(max_depth=8, n_estimators=200, random_state=2)
gb.fit(X, y)

sort_idx = X.flatten().argsort()
plt.figure(figsize=(10, 8))
plt.scatter(X[sort_idx], y[sort_idx])
plt.plot(X[sort_idx], gb.predict(X[sort_idx]), color='k')

plt.xlabel('HR')
plt.ylabel('CNT')


'''
#
Back to the main point
'''
X = bike.drop('cnt', axis='columns')
y = bike['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Instantiate gb
gb = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=2)

# Train the GB regressor
# Fit gb to the training set
gb.fit(X_train, y_train)
# Predict test set labels
y_pred = gb.predict(X_test)


# Evaluate the GB regressor
from sklearn.metrics import mean_squared_error as MSE

# Compute MSE
mse_test = MSE(y_test, y_pred)

# Compute RMSE
rmse_test = mse_test ** 0.5

# Print RMSE
print("Test set RMSE of gb: {:.3f}".format(rmse_test))


###
# Stochastic Gradient Boosting (SGB)
'''
Gradient Boosting: Cons & Pros
 - GB involves an exhaustive search procedure
 - Each CART is trained to find the best split points and features.
 - May lead to CARTs using the same split points and maybe the same features.

Stochastic Gradient Boosting
 - Each tree is trained on a random subset of rows of the training data.
 - The sampled instances (40%-80% of the training set) are sampled without replacement.
 - Features are sampled (without replacement) when choosing split points
 - Result: further ensemble diversity.
 - Effect: adding further variance to the ensemble of trees.
'''

# Regression with SGB
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4, n_estimators=200, subsample=0.9,
                                 max_features=0.75, random_state=2)

# Train the SGB regressor
# Fit sgbr to the training set
sgbr.fit(X_train, y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)

# Evaluate the SGB regressor
from sklearn.metrics import mean_squared_error as MSE

# Compute test set MSE
mse_test = MSE(y_test, y_pred)

# Compute test set RMSE
rmse_test = mse_test ** 0.5

# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test)) # 47.260

