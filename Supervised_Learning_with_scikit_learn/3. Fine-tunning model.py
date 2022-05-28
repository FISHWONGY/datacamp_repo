from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

####################################
# CONFUSION MATRIX & F1 score

df = pd.read_csv('./Supervised Learning with scikit-learn/data/house-votes-84.csv')
df.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
              'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
              'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df.replace({'?': 'n'}, inplace=True)
df.replace({'n': 0, 'y':  1}, inplace=True)
y = df['party'].values
X = df.drop('party', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# try writing a loop to test
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 25)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Now answer the question why 8 from line 81, plot shows sweet spot around 8-9, focus on testing accuracy
# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Confusion matrix
print(confusion_matrix(y_test, y_pred))

# Result of the different metric
print(classification_report(y_test, y_pred))


###############################################
# ROC curve
# Logistic regression for binary classification
# still using voteing dataset
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# New stuff
# Obtain the probability of our log weight model before using a threshold to predict the label
# predict_proba - Returns array with 2 cols. Each col contains the probability for the resprctive target value
# 即係每個row有幾多percent會係democrat (0), 幾多percent會係 repunlicant (1)
# This is for republicant
df = pd.read_csv('./Supervised Learning with scikit-learn/data/house-votes-84.csv')
df.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
              'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
              'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df.replace({'?': 'n'}, inplace=True)
df.replace({'n': 0, 'y':  1}, inplace=True)
df.replace({'democrat': 0, 'republican':  1}, inplace=True)
y = df['party'].values
X = df.drop('party', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# fpr = false positive rate, fpr = true positive rate, y_test = actual label, y_pred_prob = the predicted probability
# roc_curve requires 'y', y_test to be a binanry numeric number
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot the fpr and tpr
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC curve')
plt.show()


# Arean under ROC curve
# The larger the area under the ROC curve, the better the model (AUC - area under curve)
df = pd.read_csv('./Supervised Learning with scikit-learn/data/house-votes-84.csv')
df.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
              'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
              'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df.replace({'?': 'n'}, inplace=True)
df.replace({'n': 0, 'y':  1}, inplace=True)
y = df['party'].values
X = df.drop('party', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
# roc_auc_score can use categorical variable, as long as its binary
roc_auc_score(y_test, y_pred_prob)

# AUC using cross-validation
cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')


# Hyperparameter Tunning
# Grid search cross-validation
df = pd.read_csv('/Online course/datacamp/Supervised Learning with scikit-learn/data/house-votes-84.csv')
df.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
              'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
              'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df.replace({'?': 'n'}, inplace=True)
df.replace({'n': 0, 'y':  1}, inplace=True)
y = df['party'].values
X = df.drop('party', axis=1).values

# Specify a hyperparameter - a dictionary
# n_nrighbors for KNN / alpha for ridge & lasso
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
# perform that actual grid searching process
knn_cv.fit(X, y)

print(knn_cv.best_params_)

print(knn_cv.best_score_)


# EXERCISE 1
df = pd.read_csv('./Supervised Learning with scikit-learn/data/diabetes.csv')

"""X = df.drop('pregnancies', axis=1).values
y = df['pregnancies'].values"""

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

logreg = LogisticRegression(max_iter=1000)
# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
# Fit it to the data
logreg_cv.fit(X, y)

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


# EXERCISE 2
# Hyperparameter tuning with RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# Hold-out set for final evaluation
# given the score of the function of choice,
# might want to report on how well I will expect my model to perform on data never seen before
# Thus, split data into training set and hold-out set

df = pd.read_csv('./Supervised Learning with scikit-learn/data/diabetes.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Hold-out set in practice I: Classification

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression(max_iter=1000, solver='liblinear')

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


# Hold-out set in practice II: Regression
df = pd.read_csv('/Online course/datacamp/Supervised Learning with scikit-learn/data/gm_2008_region.csv')
# Region is str, needs to change to int instead
region_dict = {
    'Middle East & North Africa': 0,
    'Sub-Saharan Africa': 1,
    'America': 2,
    'Europe & Central Asia': 3,
    'East Asia & Pacific': 4,
    'South Asia': 5
}

df['Region'] = df['Region'].map(region_dict)
# Create arrays for features and target variable
X = df.drop('life', axis='columns').values
y = df['life'].values

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet(max_iter=1000, tol=0.001)

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))




