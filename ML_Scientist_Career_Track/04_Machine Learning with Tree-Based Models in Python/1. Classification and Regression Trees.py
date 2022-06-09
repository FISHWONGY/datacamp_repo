import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Decision tree for classification
'''
Classification-tree
 - Sequence of if-else questions about individual features.
 - Objective: infer class labels
 - Able to caputre non-linear relationships between features and labels
 - Don't require feature scaling(e.g. Standardization)

Decision Regions
 - Decision region: region in the feature space where all instances are assigned to one class label
 - Decision Boundary: surface separating different decision regions
'''

# Train your first classification tree
wbc = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                  '04_Machine Learning with Tree-Based Models in Python/data/wbc.csv')

X = wbc[['radius_mean', 'concave points_mean']]
y = wbc['diagnosis']
y = y.map({'M': 1, 'B': 0})

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth=6, random_state=1)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
print(y_pred[0:5])


# Evaluate the classification tree
# Import accuracy_score
from sklearn.metrics import accuracy_score

# Predict test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))

# Logistic regression vs classification tree
'''
A classification tree divides the feature space into rectangular regions. 
In contrast, a linear model such as logistic regression produces only a single linear decision 
boundary dividing the feature space into two decision regions.
'''

# Helper function
'''Function producing a scatter plot of the instances contained
   in the 2D dataset (X,y) along with the decision
   regions of two trained classification models contained in the
   list 'models'.

   Parameters
   ----------
   X: pandas DataFrame corresponding to two numerical features
   y: pandas Series corresponding the class labels
   models: list containing two trained classifiers
'''
from mlxtend.plotting import plot_decision_regions


def plot_labeled_decision_regions(X, y, models):
    if len(models) != 2:
        raise Exception('''Models should be a list containing only two trained classifiers.''')
    if not isinstance(X, pd.DataFrame):
        raise Exception('''X has to be a pandas DataFrame with two numerical features.''')
    if not isinstance(y, pd.Series):
        raise Exception('''y has to be a pandas Series corresponding to the labels.''')
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 5), sharey=True)
    for i, model in enumerate(models):
        plot_decision_regions(X.values, y.values, model, legend=2, ax=ax[i])
        ax[i].set_title(model.__class__.__name__)
        ax[i].set_xlabel(X.columns[0])
        if i == 0:
            ax[i].set_ylabel(X.columns[1])
            ax[i].set_ylim(X.values[:, 1].min(), X.values[:, 1].max())
            ax[i].set_xlim(X.values[:, 0].min(), X.values[:, 0].max())
    plt.tight_layout()


from sklearn.linear_model import LogisticRegression

# Instantiate logreg
logreg = LogisticRegression(random_state=1)

# Fit logreg to the training set
logreg.fit(X_train, y_train)

# Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]

# Review the decision regions of the two classifier
plot_labeled_decision_regions(X_test, y_test, clfs)


# Classification tree Learning
'''
Building Blocks of a Decision-Tree
 - Decision-Tree: data structure consisting of a hierarchy of nodes
 - Node: question or prediction
 - Three kinds of nodes
    - Root: no parent node, question giving rise to two children nodes.
    - Internal node: one parent node, question giving rise to two children nodes.
    - Leaf: one parent node, no children nodes --> prediction.

 - Criteria to measure the impurity of a note :
    - gini index
    - entropy

- Classification-Tree Learning
 - Nodes are grown recursively.
 - At each node, split the data based on:
    - feature  and split-point  to maximize
'''

# Using entropy as a criterion
from sklearn.tree import DecisionTreeClassifier

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)


# Entropy vs Gini index
dt_gini = DecisionTreeClassifier(max_depth=8, criterion='gini', random_state=1)
dt_gini.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Use dt_entropy to predict test set labels
y_pred = dt_entropy.predict(X_test)
y_pred_gini = dt_gini.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)
accuracy_gini = accuracy_score(y_test, y_pred_gini)

# Print accuracy_entropy
print("Accuracy achieved by using entropy: ", accuracy_entropy)

# Print accuracy_gini
print("Accuracy achieved by using gini: ", accuracy_gini)

# Entropy perform better in this case

# Decision tree for regression
mpg = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                  '04_Machine Learning with Tree-Based Models in Python/data/auto.csv')

X = mpg.drop('mpg', axis='columns')
y = mpg['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

from sklearn.tree import DecisionTreeRegressor

# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)

# Fit dt to the training set
dt.fit(X_train, y_train)


# Evaluate the regression tree
from sklearn.metrics import mean_squared_error

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = mean_squared_error(y_test, y_pred)

# Compute rmse_dt
rmse_dt = mse_dt ** (1/2)

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))


# Linear regression vs regression tree
# Preprocess
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

# Predict test set labels
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
mse_lr = mean_squared_error(y_test, y_pred_lr)

# Compute rmse_lr
rmse_lr = mse_lr ** 0.5

# Print rmse_lr
print("Linear Regression test set RMSE: {:.2f}".format(rmse_lr))

# Print rmse_dt
print("Regression Tree test set RMSE: {:.2f}".format(rmse_dt))

