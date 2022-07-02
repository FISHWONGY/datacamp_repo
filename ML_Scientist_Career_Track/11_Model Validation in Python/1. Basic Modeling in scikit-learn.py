import pandas as pd
import numpy as np

# Introduction to model validation
'''
Model validation
    - Ensuring your model performs as expected on new data
    - Testing model performance on holdout datasets
    - Selecting the best model, parameters, and accuracy metrics
    - Achieving the best accuracy for the given data

# Seen vs. unseen data
Model's tend to have higher accuracy on observations they have seen before. 
In the candy dataset, predicting the popularity of Skittles will likely have higher accuracy than 
predicting the popularity of Andes Mints; Skittles is in the dataset, and Andes Mints is not.

You've built a model based on 50 candies using the dataset X_train and need to report how accurate 
the model is at predicting the popularity of the 50 candies the model was built on, 
and the 35 candies (X_test) it has never seen. You will use the mean absolute error, mae(), as the accuracy metric.
'''

candy = pd.read_csv('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                    '11_Model Validation in Python/data/candy-data.csv')
print(candy.head())

X = candy.drop(['competitorname', 'winpercent'], axis=1)
y = candy['winpercent']


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = RandomForestRegressor(n_estimators=50)

# The model is fit using X_train and y_train
model.fit(X_train, y_train)

# Create vectors of predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Train/Test Errors
train_error = mae(y_true=y_train, y_pred=train_predictions)
test_error = mae(y_true=y_test, y_pred=test_predictions)

# Print the accuracy for seen and unseen data
print("Model error on seen data: {0:.2f}.".format(train_error))
print("Model error on unseen data: {0:.2f}.".format(test_error))

# Regression models
'''
Random forest parameters
    n_estimators: the number of trees in the forest
    max_depth: the maximum depth of the trees
    random_state: random seed

Set parameters and fit a model
Predictive tasks fall into one of two categories: regression or classification. 
In the candy dataset, the outcome is a continuous variable describing how often the candy was chosen 
over another candy in a series of 1-on-1 match-ups. To predict this value (the win-percentage), 
you will use a regression model.
'''

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

# Set the number of trees
rfr.n_estimators = 100

# Add a maximum depth
rfr.max_depth = 6

# Set the random date
rfr.random_state = 1111

# Fit the model
rfr.fit(X_train, y_train)

# You have updated parameters after the model was initialized.
# This approach is helpful when you need to update parameters.
# Before making predictions, let's see which candy characteristics were most important to the model.

# Feature importances
'''
Although some candy attributes, such as chocolate, may be extremely popular, 
it doesn't mean they will be important to model prediction. 
After a random forest model has been fit, you can review the model's attribute, .feature_importances_, 
to see which variables had the biggest impact. 
You can check how important each variable was in the model by looping over 
the feature importance array using enumerate().
'''

# Print how important each column is to the model
for i, item in enumerate(rfr.feature_importances_):
    # Use i and item to print out the feature importance of each column
    print("{0:s}: {1:.2f}".format(X_train.columns[i], item))


# Classification models
'''
Classification predictions

In model validation, it is often important to know more about the predictions than just the final classification. 
When predicting who will win a game, most people are also interested in how likely it is a team will win.

In this exercise, you look at the methods, .predict() and .predict_proba() using the tic_tac_toe dataset. 
The first method will give a prediction of whether Player One will win the game, and the second method will 
provide the probability of Player One winning.
'''
tic_tac_toe = pd.read_csv('./Online course/datacamp_repo/ML_Scientist_Career_Track/'
                          '11_Model Validation in Python/data/tic-tac-toe.csv')
print(tic_tac_toe.head())

y = tic_tac_toe['Class'].apply(lambda x: 1 if x == 'positive' else 0)
X = tic_tac_toe.drop('Class', axis=1)
X = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
rfc = RandomForestClassifier()

# Fit the rfc model
rfc.fit(X_train, y_train)

# Create arrays of predictions
classification_predictions = rfc.predict(X_test)
probability_predictions = rfc.predict_proba(X_test)

# Print out count of binary predictions
print(pd.Series(classification_predictions).value_counts())

# Print the first value from probability_predictions
print('The first predicted probabilities are: {}'.format(probability_predictions[0]))

# You can see there were 563 observations where Player One was predicted to win the Tic-Tac-Toe game.
# Also, note that the predicted_probabilities array contains lists with only two values because you only
# have two possible responses (win or lose).
# Remember these two methods, as you will use them a lot throughout this course.

# Reusing model parameters
'''
Replicating model performance is vital in model validation. Replication is also important when sharing models with co-workers, reusing models on new data or asking questions on a website such as Stack Overflow. You might use such a site to ask other coders about model errors, output, or performance. The best way to do this is to replicate your work by reusing model parameters.

In this exercise, you use various methods to recall which parameters were used in a model.
'''


rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Print the classification model
print(rfc)

# Print the classification model's random state parameter
print('The random state is: {}'.format(rfc.random_state))

# Print all parameters
print('Printing the parameters dictionary: {}'.format(rfc.get_params()))

# Recalling which parameters were used will be helpful going forward. Model validation and performance rely
# heavily on which parameters were used, and there is no way to replicate a model without
# keeping track of the parameters used!


# Random forest classifier
'''
This exercise reviews the four modeling steps discussed throughout this chapter using a random forest classification model. You will:
    Create a random forest classification model.
    Fit the model using the tic_tac_toe dataset.
    Make predictions on whether Player One will win (1) or lose (0) the current game.
    Finally, you will evaluate the overall accuracy of the model.
'''


# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Fit rfc using X_train and y_train
rfc.fit(X_train, y_train)

# Create predictions on X_test
predictions = rfc.predict(X_test)
print(predictions[0:5])

# Print model accuracy using score() and the testing data
print(rfc.score(X_test, y_test))

