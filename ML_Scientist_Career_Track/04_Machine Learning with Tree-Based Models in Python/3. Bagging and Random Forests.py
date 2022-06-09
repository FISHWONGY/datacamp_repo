import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bagging
'''
Ensemble Methods
 - Voting Classifier
    - same training set,
    - NOT algortihms
 - Bagging
    - One algorithm
    - NOT subsets of the training set

Bagging
 - Bootstrap Aggregation
 - Uses a technique known as the bootstrap
 - Reduces variance of individual models in the ensemble _ Bootstrap
'''

'''
Practical - 
Classification: BaggingClassifier in scikit-learn
Aggreagtes predictions by majority voting

Regression: BaggingRegressor in scikit-learn
Aggreagtes predictions through averaging
'''

# Define the bagging classifier
indian = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                     '04_Machine Learning with Tree-Based Models in Python/data/'
                     'indian_liver_patient_preprocessed.csv', index_col=0)

# X - df; y - Series
X = indian.drop('Liver_disease', axis='columns')
y = indian['Liver_disease']

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)


# Evaluate Bagging performance
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

from sklearn.metrics import accuracy_score

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bc: {:.2f}'.format(acc_test))

# Evaluate using cross_val_score using K-fold Cross-Validation
# Similar result 66-70%
from sklearn.model_selection import cross_val_score
cross_val_score(bc, X_train, y_train, cv=3, scoring='accuracy')

# Confusion Matrix
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(bc, X_train, y_train, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_train_pred)

import pandas as pd
pd.DataFrame(confusion_matrix(y_train, y_train_pred))

pd.DataFrame(confusion_matrix(y_train, y_train_pred),
             columns=pd.MultiIndex.from_product([['Prediction'], ["Negative", "Positive"]]),
             index=pd.MultiIndex.from_product([["Actual"], ["Negative", "Positive"]]))

# Precision
'''
precision = True Positives / True Positives + False Positives
'''
from sklearn.metrics import precision_score, recall_score
precision_score(y_train, y_train_pred)
# Recall
'''
recall = True Positives / True Positives + False Negatives
'''
recall_score(y_train, y_train_pred)

# F1 Score
# 𝐹1 score is the harmonic mean of precision and recall. Regular mean gives equal weight to all values.
# Harmonic mean gives more weight to low values.
from sklearn.metrics import f1_score
f1_score(y_train, y_train_pred)

# Try using the Decision Tree model
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

acc_test_dt = accuracy_score(y_test, y_pred_dt)
print('Test set accuracy of dt: {:.2f}'.format(acc_test_dt))


# Visualizing features importances
# Create a pd.Series of features importances
importances = pd.Series(data=dt.feature_importances_, index=X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
plt.figure(figsize=(12, 8))
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

# Out of Bag Evaluation
'''
Bagging
 - Some instances may be sampled several times for one model, other instances may not be sampled at all.

Out Of Bag (OOB) instances
 - On average, for each model, 63% of the training instances are sampled
 - The remaining 37% constitute the OOB instances
'''

# Prepare the ground
'''
In sklearn, you can evaluate the OOB accuracy of an ensemble classifier by setting the parameter 
oob_score to True during instantiation. After training the classifier, 
the OOB accuracy can be obtained by accessing the .oob_score_ attribute from the corresponding instance.
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, oob_score=True, random_state=1)

# OOB Score vs Test Set Score
# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_test, y_pred)
cross_val_score(bc, X_train, y_train, cv=3, scoring='accuracy')

# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))


###
# Random Forests (RF)
'''
Bagging
 - Base estimator: Decision Tree, Logistic Regression, Neural Network, ...
 - Each estimator is trained on a distinct bootstrap sample of the training set
 - Estimators use all features for training and prediction

Further Diversity with Random Forest
 - Base estimator: Decision Tree
 - Each estimator is trained on a different bootstrap sample having the same size as the training set
 - RF introduces further randomization in the training of individual trees
 - features are sampled at each node without replacement

Feature importance
 - Tree based methods: enable measuring the importance of each feature in prediction
'''

# Train an RF regressor
bike = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                   '04_Machine Learning with Tree-Based Models in Python/data/bikes.csv')

# X - DF; y - Series
X = bike.drop('cnt', axis='columns')
y = bike['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25, random_state=2)

# Fit rf to the training set
rf.fit(X_train, y_train)


# Evaluate the RF regressor
from sklearn.metrics import mean_squared_error as MSE

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** 0.5

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))


# Visualizing features importances
# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_, index=X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
plt.figure(figsize=(12, 8))
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')

# Apparently, hr and workingday are the most important features according to rf.
# The importances of these two features add up to more than 90%!

