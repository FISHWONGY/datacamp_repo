import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## Introduction
- Parameters
    - Components of the model learned during the modeling process
    - Do not set these manually

### Extracting a Logistic Regression parameter
You are now going to practice extracting an important parameter of the logistic regression model. The logistic regression has a few other parameters you will not explore here but you can review them in the [scikit-learn.org](https://scikit-learn.org/) documentation for the `LogisticRegression()` module under 'Attributes'.

This parameter is important for understanding the direction and magnitude of the effect the variables have on the target.

In this exercise we will extract the coefficient parameter (found in the `coef_` attribute), zip it up with the original column names, and see which variables had the largest positive effect on the target variable.
"""

credit_card = pd.read_csv('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                          '21_Hyperparameter Tuning in Python/data/credit-card-full.csv')
# To change categorical variable with dummy variables
credit_card = pd.get_dummies(credit_card, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)
credit_card.head()

from sklearn.model_selection import train_test_split

X = credit_card.drop(['ID', 'default payment next month'], axis=1)
y = credit_card['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

from sklearn.linear_model import LogisticRegression

log_reg_clf = LogisticRegression(max_iter=1000)
log_reg_clf.fit(X_train, y_train)

# Create a list of original variable names from the training DataFrame
original_variables = X_train.columns

# Extract the coefficients of the logistic regression estimator
model_coefficients = log_reg_clf.coef_[0]

# Create a dataframe of the variables and coefficients & print it out
coefficient_df = pd.DataFrame({'Variable': original_variables, 
                               'Coefficient': model_coefficients})
print(coefficient_df)

# Print out the top 3 positive variables
top_three_df = coefficient_df.sort_values(by='Coefficient', axis=0, ascending=False)[0:3]
print(top_three_df)

"""### Extracting a Random Forest parameter
You will now translate the work previously undertaken on the logistic regression model to a random forest model. A parameter of this model is, for a given tree, how it decided to split at each level.

This analysis is not as useful as the coefficients of logistic regression as you will be unlikely to ever explore every split and every tree in a random forest model. However, it is a very useful exercise to peak under the hood at what the model is doing.

In this exercise we will extract a single tree from our random forest model, visualize it and programmatically extract one of the splits.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import os
import pydot

rf_clf = RandomForestClassifier(max_depth=4, criterion='gini', n_estimators=10);
rf_clf.fit(X_train, y_train)

# Extract the 7th (index 6) tree from the random forest
chosen_tree = rf_clf.estimators_[6]

# Export with dot 
export_graphviz(chosen_tree,
                out_file='./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                         '21_Hyperparameter Tuning in Python/tree6.dot',
                feature_names=X_train.columns,
                filled=True,
                rounded=True)
(graph, ) = pydot.graph_from_dot_file('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                                      '21_Hyperparameter Tuning in Python/tree6.dot')

# Convert dot to png
graph.write_png('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                '21_Hyperparameter Tuning in Pythontree_viz_image.png')

# Visualize the graph using the provided image
tree_viz_image = plt.imread('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                            '21_Hyperparameter Tuning in Pythontree_viz_image.png')
plt.figure(figsize=(16, 10))
plt.imshow(tree_viz_image, aspect='auto');
plt.axis('off')

# Extract the parameters and level of the top (index 0) node
split_column = chosen_tree.tree_.feature[0]
split_column_name = X_train.columns[split_column]
split_value = chosen_tree.tree_.threshold[0]

# Print out the feature and level
print('This node split on feature {}, at a value of {}'.format(split_column_name, split_value))

"""## Introducing Hyperparameters
- Hyperparameters
    - Something you set before the modelling process (need to tune)
    - The algorithm does not learn these

### Exploring Random Forest Hyperparameters
Understanding what hyperparameters are available and the impact of different hyperparameters is a 
core skill for any data scientist. As models become more complex, there are many different settings you can set, 
but only some will have a large impact on your model.

You will now assess an existing random forest model (it has some bad choices for hyperparameters!) 
and then make better choices for a new random forest model and assess its performance.
"""

from sklearn.metrics import confusion_matrix, accuracy_score

rf_clf_old = RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, 
                                    n_estimators=5, oob_score=False, random_state=42)

rf_clf_old.fit(X_train, y_train)
rf_old_predictions = rf_clf_old.predict(X_test)

# Print out the old estimator, notice which hyperparameter is badly set
print(rf_clf_old)

# Get confusion matrix & accuracy for the old rf_model
print('Confusion Matrix: \n\n {} \n Accuracy Score: \n\n {}'.format(
    confusion_matrix(y_test, rf_old_predictions),
    accuracy_score(y_test, rf_old_predictions)
))

# Create a new random forest classifier with better hyperparameters
rf_clf_new = RandomForestClassifier(n_estimators=500)

# Fit this to the data and obtain predictions
rf_new_predictions = rf_clf_new.fit(X_train, y_train).predict(X_test)

# Assess the new model (using new predictions!)
print('Confusion Matrix: \n\n', confusion_matrix(y_test, rf_new_predictions))
print('Accuracy Score: \n\n', accuracy_score(y_test, rf_new_predictions))

"""### Hyperparameters of KNN
To apply the concepts learned in the prior exercise, it is good practice to try out learnings on a new algorithm. The k-nearest-neighbors algorithm is not as popular as it used to be but can still be an excellent choice for data that has groups of data that behave similarly. Could this be the case for our credit card users?

In this case you will try out several different values for one of the core hyperparameters for the knn algorithm and compare performance.


"""

from sklearn.neighbors import KNeighborsClassifier

# Build a knn estimator for each value of n_neighbors
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_20 = KNeighborsClassifier(n_neighbors=20)

# Fit each to the training data & produce predictions
knn_5_predictions = knn_5.fit(X_train, y_train).predict(X_test)
knn_10_predictions = knn_10.fit(X_train, y_train).predict(X_test)
knn_20_predictions = knn_20.fit(X_train, y_train).predict(X_test)

# Get an accuracy score for each of the models
knn_5_accuracy = accuracy_score(y_test, knn_5_predictions)
knn_10_accuracy = accuracy_score(y_test, knn_10_predictions)
knn_20_accuracy = accuracy_score(y_test, knn_20_predictions)
print('The accuracy of 5, 10, 20 neighbors was {}, {}, {}'.format(knn_5_accuracy,
                                                                  knn_10_accuracy,
                                                                  knn_20_accuracy))

"""## Setting & Analyzing Hyperparameter Values

### Automating Hyperparameter Choice
Finding the best hyperparameter of interest without writing hundreds of lines of code for hundreds of models is 
an important efficiency gain that will greatly assist your future machine learning model building.

An important hyperparameter for the GBM algorithm is the learning rate. But which learning rate is best for 
this problem? By writing a loop to search through a number of possibilities, collating these and 
viewing them you can find the best one.

Possible learning rates to try include 0.001, 0.01, 0.05, 0.1, 0.2 and 0.5
"""

from sklearn.ensemble import GradientBoostingClassifier

# Set the learning rates & results storage
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
results_list = []

# Create the for loop to evaluate model predictions for each learning rate
for learning_rate in learning_rates:
    model = GradientBoostingClassifier(learning_rate=learning_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Save the learning rate and accuracy score
    results_list.append([learning_rate, accuracy_score(y_test, predictions)])
    
# Gather everything into a DataFrame
results_df = pd.DataFrame(results_list, columns=['learning_rate', 'accuracy'])
print(results_df)
# Learning rate - 0.05 is the best


"""
### Building Learning Curves
If we want to test many different values for a single hyperparameter it can be difficult to 
easily view that in the form of a DataFrame. Previously you learned about a nice trick to analyze this. 
A graph called a 'learning curve' can nicely demonstrate the effect of increasing or decreasing a 
particular hyperparameter on the final result.

Instead of testing only a few values for the learning rate, you will test many to easily see the 
effect of this hyperparameter across a large range of values. A useful function from NumPy is
 `np.linspace(start, end, num)` which allows you to create a number of values (`num`) 
 evenly spread within an interval (`start`, `end`) that you specify.
"""

# Set the learning rates & accuracies list
learn_rates = np.linspace(0.01, 2, num=30)
accuracies = []

# Create the for loop
for learn_rate in learn_rates:
    # Create the model, predictions & save the accuracies as before
    model = GradientBoostingClassifier(learning_rate=learn_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))
    
# Plot results
plt.plot(learn_rates, accuracies)
plt.gca().set(xlabel='learning_rate', ylabel='Accuracy', title='Accuracy for different learning_rates');
# Learning rate - 1.38 is the best

"""
You can see that for low values, you get a pretty good accuracy. 
However once the learning rate pushes much above 1.5, the accuracy starts to drop.
"""