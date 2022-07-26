import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""To tune the model we have to modify the hyper parameters. 
Here is how to see the hyperparameter values for the svc function"""

from sklearn.svm import SVC

svc = SVC()

str(svc)

"""One way to search the best hyperparameter is to find it using grid search """

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators': np.arange(10, 51)}

clf_cv = GridSearchCV(RandomForestClassifier(), param_grid)

clf_cv.fit(X, y)

clf_cv.best_params_

# {'n_estimators': 42}

clf_cv.best_score_

"""**Multiple Hyperparametrization**

"""

# Create the hyperparameter grid
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

"""**Randomized Search**

As the hyperparameter grid gets larger, grid search becomes slower. In order to solve this problem, instead of trying out every single combination of values, we could randomly jump around the grid and try different combinations. There's a small possibility we may miss the best combination, but we would save a lot of time, or be able to tune more hyperparameters in the same amount of time.
"""

# Import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Create the hyperparameter grid
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Call RandomizedSearchCV
random_search = RandomizedSearchCV(clf, param_dist)

# Fit the model
random_search.fit(X, y)

# Print best parameters
print(random_search.best_params_)

#{'bootstrap': True, 'criterion': 'entropy', 'max_depth': None, 'max_features': 10}

"""## Feature Importance

- Scores representing how much each feature contributes to a prediction
- Effective way to communicate results to stakeholders
    - Which features ra important drivers of churn
    - Which features can be removed from the model
"""

random_forest = RandomForestClassifier()

random_forest.fit(X_train, y_train)

print(random_forest.feature_importances_)

# Sort importances
sorted_index = np.argsort(importances)

# Create labels
labels = X.columns[sorted_index]

# Clear current plot
plt.clf()

# Create plot
plt.barh(range(X.shape[1]), importances[sorted_index], tick_label=labels)
plt.show()

"""## Adding new Features

Adding new features can help to improve the model. Let's see an example

Let's consider that for the Dataset the following features were added:
- Region_Code
- Cost_Call
- Total_Charge
- Total_Minutes
- Total_Calls
- Min_Call

"""

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Instantiate the classifier
clf = RandomForestClassifier()

# Fit to the data
clf.fit(X_train, y_train)

# Print the accuracy
print(clf.score(X_test, y_test))
