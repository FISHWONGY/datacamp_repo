import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
df = pd.read_csv('/Online course/datacamp/Supervised Learning with scikit-learn/data/auto.csv')

df_origin = pd.get_dummies(df)
df_origin = df_origin.drop('origin_Asia', axis=1)

# Create arrays for features and target variable
X = df_origin.drop('mpg', axis='columns').values
y = df_origin['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
ridge = Ridge(alpha=0.5, normalize=True).fit(X_train, y_train)
ridge.score(X_test, y_test)


# Exercise 1
df = pd.read_csv('./Supervised Learning with scikit-learn/'
                 'data/gm_2008_region.csv')
# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)
plt.show()

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)

# Regression with categorical features
'''
# Having created the dummy variables from the 'Region' feature, 
# you can build regression models as you did before. 
# Here, you'll use ridge regression to perform 5-fold cross-validation.
'''
# Preprocess
X = df_region.drop('life', axis='columns')
y = df_region['life']

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)

# Handling missing data
'''
Dropping missing data
It can remove most of datas, we need a more robust method.
Imputing missing data
Making an educated guess about the missing values
Example : Using the mean of the non-missing entries
'''

# Exercise 1
# Preprocessing

df = pd.read_csv('./Supervised Learning with scikit-learn/'
                 'data/house-votes-84.csv', header=None)
df.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
              'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
              'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df.replace({'n': 0, 'y': 1}, inplace=True)

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print('Shape of Original DataFrame: {}'.format(df.shape))

# Drop missing values and print shape fo new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

# Imputing missing data in a ML Pipeline I
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
         ('SVM', clf)]


# Imputing missing data in a ML Pipeline II
# Having setup the steps of the pipeline in the previous exercise,
# you will now use it on the voting dataset to classify a Congressman's party affiliation.
# What makes pipelines so incredibly useful is the simple interface that they provide.
# You can use the .fit() and .predict() methods on pipelines just as you did with your classifiers and regressors!

df = pd.read_csv('./Supervised Learning with scikit-learn/'
                 'data/house-votes-84.csv', header=None)
df.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
              'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
              'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df.replace({'?': 'n'}, inplace=True)
df.replace({'n': 0, 'y':  1}, inplace=True)

# Preprocessing
X = df.drop('party', axis='columns')
y = df['party']


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
         ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))

# Centering and scaling
'''
Why scale your data?
Many models use some form of distance to inform them
Features on larger scales can unduly influence the model
Example: k-NN uses distance explicitly when making predictions
Want features to be on a similar scale
Normalizing (or scaling and centering)
Ways to normalize your data
Standardization: Subtract the mean and divide by variance
All features are centered around zero and have variance one
Can also subtract the minimum and divide by the range
Minimum zero and maximum one
Can also normalize so the data ranges from -1 to +1
'''

#  Preprocess
df = pd.read_csv('./Supervised Learning with scikit-learn/'
                 'data/white-wine.csv')

df['quality'] = df['quality'] < 5
X = df.drop('quality', axis='columns').values
y = df['quality'].values
df.head()

from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X)))
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

# Centering and scaling in a pipeline
# With regard to whether or not scaling is effective, the proof is in the pudding!
# See for yourself whether or not scaling the features of the White Wine Quality dataset
# has any impact on its performance.
# You will use a k-NN classifier as part of a pipeline that includes scaling,
# and for the purposes of comparison, a k-NN classifier trained on the unscaled data has been provided.

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

# Bringing it all together I: Pipeline for classification

# It is time now to piece together everything you have learned so far into a pipeline for classification!
# Your job in this exercise is to build a pipeline that includes scaling
# and hyperparameter tuning to classify wine quality.
# You'll return to using the SVM classifier you were briefly introduced to earlier in this chapter.
# The hyperparameters you will tune are C and gamma. C controls the regularization strength.
# It is analogous to the C you tuned for logistic regression in Chapter 3,
# while gamma controls the kernel coefficient: Do not worry about this now as it is beyond the scope of this course.

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


# Bringing it all together II: Pipeline for regression

# For this final exercise, you will return to the Gapminder dataset. Guess what?
# Even this dataset has missing values that we dealt with for you in earlier chapters!
# Now, you have all the tools to take care of them yourself!
# Your job is to build a pipeline that imputes the missing data,
# scales the features, and fits an ElasticNet to the Gapminder data.
# You will then tune the l1_ratio of your ElasticNet using GridSearchCV.

df = pd.read_csv('./Supervised Learning with scikit-learn/'
                 'data/gm_2008_region.csv')
df.drop(['Region'], axis='columns', inplace=True)
X = df.drop('life', axis='columns').values
y = df['life'].values

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='mean')),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet(tol=0.6))]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio': np.linspace(0, 1, 30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
