import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

plt.rcParams['figure.figsize'] = (10, 5)

# Introduction
'''
Supervised Learning
 - Relies on labeled data
 - Have some understanding of past behavior

**IMPORTANT**
AUC: Metric for binary classification models
 - Area Under the ROC Curve (AUC)
    - Larger area under the ROC curve = better model

Other supervised learning considerations
 - Features can be either numeric or categorical
 - Numeric features should be scaled (Z-scored)
 - Categorical features should be encoded (one-hot)
'''

# Introducing XGBoost
'''
What is XGBoost? (eXtreme Gradient Boosting)
 - Optimized gradient-boosting machine learning library
 - Originally written in C++
 - Has APIs in several languages: Python, R, Scala, Julia, Java

What makes XGBoost so popular?
 - Speed and performance
 - Core algorithm is parallelizable
 - Consistently outperforms single-algorithm methods
 - State-of-the-art performance in many ML tasks
'''

# XGBoost - Fit/Predict
churn_data = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                         '05_Extreme Gradient Boosting with XGBoost/data/churn_data.csv')

from sklearn.model_selection import train_test_split

# Create arrays for the features and the target: X - DF, y - Series
X, y = churn_data.iloc[:, :-1], churn_data.iloc[:, -1]

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
print("accuracy: %f" % (accuracy))


# Which features most important?
result = pd.DataFrame(xg_cl.feature_importances_, X.columns)
result.columns = ['feature']
result = result.sort_values(by='feature', ascending=False)
result.sort_values(by='feature', ascending=False).plot(kind='bar')


###
# What is a decision tree?
'''
Decision trees as base learners
 - Base learner : Individual learning algorithm in an ensemble algorithm
 - Composed of a series of binary questions
 - Predictions happen at the "leaves" of the tree

CART: Classification And Regression Trees
 - Each leaf always contains a real-valued score
 - Can later be converted into categories
'''

# X - array; y - array
X = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                '05_Extreme Gradient Boosting with XGBoost/data/breast_X.csv').to_numpy()
y = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                '05_Extreme Gradient Boosting with XGBoost/data/breast_y.csv').to_numpy().ravel()

from sklearn.tree import DecisionTreeClassifier

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4 == y_test)) / y_test.shape[0]
print("Accuracy:", accuracy)


# What is Boosting?
'''
Boosting overview
 - Not a specific machine learning algorithm
 - Concept that can be applied to a set of machine learning models
    - "Meta-algorithm"
 - Ensemble meta-algorithm used to convert many weak learners into a strong learner

Weak learners and strong learners
 - Weak learner: ML algorithm that is slightly better than chance
 - Boosting converts a collection of weak learners into a strong learner
 - Strong learner: Any algorithm that can be tuned to achieve good performance.

How boosting is accomplished?
 - Iteratively learning a set of week models on subsets of the data
 - Weighting each weak prediction according to each weak learner's performance
 - Combine the weighted predictions to obtain a single weighted prediction
 - that is much better than the individual predictions themselves!

Model evaluation through cross-validation
 - Cross-validation: Robust method for estimating the performance of a model on unseen data
 - Generates many non-overlapping train/test splits on training data
 - Reports the average test set performance across all data splits
'''

# Measuring accuracy
'''
XGBoost gets its lauded performance and efficiency gains by utilizing its own optimized data structure 
for datasets called a DMatrix

when you use the xgboost cv object, you have to first explicitly convert your data into a DMatrix. 
So, that's what you will do here before running cross-validation on churn_data
'''

churn_data = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                         '05_Extreme Gradient Boosting with XGBoost/data/churn_data.csv')

# Create arrays for the features and the target: X - DF; y - Series
X, y = churn_data.iloc[:, :-1], churn_data.iloc[:, -1]

# Create the DMatrix from X and y, dtype -  churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {'objective': "reg:logistic", "max_depth": 3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params,
                    nfold=3, num_boost_round=5,
                    metrics="error", as_pandas=True, seed=123)

# Pint cv_results
print(cv_results)

# Print the accuracy
print(((1 - cv_results['test-error-mean']).iloc[-1]))

# cv_results stores the training and test mean and standard deviation of the error per boosting round (tree built)
# as a DataFrame. From cv_results,
# the final round 'test-error-mean' is extracted and converted into an accuracy, where accuracy is 1-error.
# The final accuracy of around 75% is an improvement from earlier!


# Measuring AUC
# Compute the common metric used in binary classification - the area under the curve ("auc")
# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params,
                    nfold=3, num_boost_round=5,
                    metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])

# An AUC of 0.84 is quite strong.
# As you have seen, XGBoost's learning API makes it very easy to compute any metric you may be interested in.
# In Chapter 3, you'll learn about techniques to fine-tune your XGBoost models to
# improve their performance even further.
# For now, it's time to learn a little about exactly when to use XGBoost.


# When should I use XGBoost?
'''
When to use XGBoost
 - You have a large number of training samples
    - Greater than 1000 training samples and less 100 features
    - The number of features < number of training samples
 - You have a mixture of categorical and numeric features
    - Or just numeric features

When to NOT use XGBoost
 - Image recognition
 - Computer vision
 - Natural language processing and understanding problems
 - When the number of training samples is significantly smaller than the number of features
'''