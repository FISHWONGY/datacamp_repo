import pandas as pd
import numpy as np
from pprint import pprint

"""## Introducing Grid Search

### Build Grid Search functions
In data science it is a great idea to try building algorithms, models and processes 'from scratch' so you can really understand what is happening at a deeper level. Of course there are great packages and libraries for this work (and we will get to that very soon!) but building from scratch will give you a great edge in your data science work.

In this exercise, you will create a function to take in 2 hyperparameters, build models and return results. You will use this function in a future exercise.
"""

from sklearn.model_selection import train_test_split

credit_card = pd.read_csv('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                          '21_Hyperparameter Tuning in Python/data/credit-card-full.csv')
# To change categorical variable with dummy variables
credit_card = pd.get_dummies(credit_card, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)

X = credit_card.drop(['ID', 'default payment next month'], axis=1)
y = credit_card['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# Create the function
def gbm_grid_search(learn_rate, max_depth):
    # Create the model
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth)
    
    # Use the model to make predictions
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Return the hyperparameters and score
    return ([learn_rate, max_depth, accuracy_score(y_test, predictions)])


"""
# Iteratively tune multiple hyperparameters
In this exercise, you will build on the function you previously created to take in 2 hyperparameters,
build a model and return the results. You will now use that to loop through some values and then 
extend this function and loop with another hyperparameter.
"""

# Create the relevant lists
results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2, 4, 6]

# Create the for loop
for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        results_list.append(gbm_grid_search(learn_rate, max_depth))
        
# Print the results
pprint(results_list)

# Extend the function input
def gbm_grid_search_extended(learn_rate, max_depth, subsample):
    # Extend the model creation section
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth,
                                       subsample=subsample)
    
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Extend the return part
    return([learn_rate, max_depth, subsample, accuracy_score(y_test, predictions)])


# Create the new list to test
subsample_list = [0.4, 0.6]

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        # Extend the for loop
        for subsample in subsample_list:
            # Extend the results to include the new hyperparameter
            results_list.append(gbm_grid_search_extended(learn_rate, max_depth, subsample))
            
# Print the results
pprint(results_list)

"""## Grid Search with Scikit Learn
- Steps in a Grid Search
    1. An algorithm to tune the hyperparameters (or estimator)
    2. Defining which hyperparameters to tune
    3. Defining a range of values for each hyperparameter
    4. Setting a cross-validatoin scheme
    5. Defining a score function so we can decide which square on our grid was 'the best'
    6. Include extra useful information or functions

### GridSearchCV with Scikit Learn
The `GridSearchCV` module from Scikit Learn provides many useful features to assist with efficiently undertaking a grid search. You will now put your learning into practice by creating a `GridSearchCV` object with certain parameters.

The desired options are:

- A Random Forest Estimator, with the split criterion as 'entropy'
- 5-fold cross validation
- The hyperparameters `max_depth` (2, 4, 8, 15) and `max_features` ('auto' vs 'sqrt')
- Use `roc_auc` to score the models
- Use 4 cores for processing in parallel
- Ensure you refit the best model and return training scores
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier with specified criterion
rf_class = RandomForestClassifier(criterion='entropy')

# Create the parametergrid
param_grid = {
    'max_depth': [2, 4, 8, 15],
    'max_features': ['auto', 'sqrt']
}

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(
    estimator=rf_class,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=4,
    cv=5,
    refit=True,
    return_train_score=True
)

print(grid_rf_class)

"""## Understanding a grid search output

### Exploring the grid search results
You will now explore the `cv_results_` property of the GridSearchCV object defined in the video. This is a dictionary that we can read into a pandas DataFrame and contains a lot of useful information about the grid search we just undertook.

A reminder of the different column types in this property:

- `time_` columns
- `param_` columns (one for each hyperparameter) and the singular params column (with all hyperparameter settings)
- a `train_score` column for each cv fold including the mean_train_score and std_train_score columns
- a `test_score` column for each cv fold including the mean_test_score and std_test_score columns
- a `rank_test_score` column with a number from 1 to n (number of iterations) ranking the rows based on their `mean_test_score`
"""

grid_rf_class.fit(X_train, y_train)

# Read the cv_results property into adataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df)

# Extract and print the column with a dictionary of hyperparameters used
column = cv_results_df.loc[:, ["params"]]
print(column)

# Extract and print the row that had the best mean test score
best_row = cv_results_df[cv_results_df['rank_test_score'] == 1]
print(best_row)

"""### Analyzing the best results
At the end of the day, we primarily care about the best performing 'square' in a grid search. Luckily Scikit Learn's `gridSearchCV` objects have a number of parameters that provide key information on just the best square (or row in `cv_results_`).

Three properties you will explore are:

- `best_score_` – The score (here ROC_AUC) from the best-performing square.
- `best_index_` – The index of the row in `cv_results_` containing information on the best-performing square.
- `best_params_` – A dictionary of the parameters that gave the best score, for example 'max_depth': 10
"""

# Print out the ROC_AUC score from the best-performing square
best_score = grid_rf_class.best_score_
print(best_score)

# Create a variable from the row related to the best-performing square
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
best_row = cv_results_df.loc[[grid_rf_class.best_index_]]
print(best_row)

# Get the max_depth parameter from the best-performing square and print
best_max_depth = grid_rf_class.best_params_['max_depth']
print(best_max_depth)

"""### Using the best results
While it is interesting to analyze the results of our grid search, our final goal is practical in nature; we want to make predictions on our test set using our estimator object.

We can access this object through the `best_estimator_` property of our grid search object.

In this exercise we will take a look inside the `best_estimator_` property and then use this to make predictions on our test set for credit card defaults and generate a variety of scores. Remember to use `predict_proba` rather than `predict` since we need probability values rather than class labels for our roc_auc score. We use a slice `[:,1]` to get probabilities of the positive class.
"""

from sklearn.metrics import confusion_matrix, roc_auc_score

# See what type of object the best_estimator_property is
print(type(grid_rf_class.best_estimator_))

# Create an array of predictions directly using the best_estimator_property
predictions = grid_rf_class.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix
print("Confusion Matrix \n", confusion_matrix(y_test, predictions))

# Get the ROC-AUC score
predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:, 1]
print("ROC-AUC Score \n", roc_auc_score(y_test, predictions_proba))

"""The `.best_estimator_` property is a really powerful property to understand for streamlining 
your machine learning model building process. You now can run a grid search and seamlessly use the 
best model from that search to make predictions."""