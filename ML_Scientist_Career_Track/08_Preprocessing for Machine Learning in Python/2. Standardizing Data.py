import pandas as pd
import numpy as np

# Standardizing Data
'''
Standardization
 - Preprocessing method used to transform continuous data to make it look normally distributed
 - Scikit-learn models assume normally distributed data
    - Log normalization
    - feature Scaling

When to standardize: models
 - Model in linear space
 - Dataset features have high variance
 - Dataset features are continuous and on different scales
 - Linearity assumptions
'''

# Modeling without normalizing
'''
Let's take a look at what might happen to your model's accuracy if you try to model data without doing some sort of 
standardization first. Here we have a subset of the wine dataset. One of the columns, Proline, 
has an extremely high variance compared to the other columns. 
This is an example of where a technique like log normalization would come in handy, 
which you'll learn about in the next section.

The scikit-learn model training process should be familiar to you at this point, so we won't go too in-depth with it. 
You already have a k-nearest neighbors model available (knn) as well as the X and y sets you need to fit and score on.
'''

wine = pd.read_csv(
    '/Online course/datacamp_repo/ML_Scientist_Career_Track/08_Preprocessing for Machine Learning in Python/data/wine_types.csv')
wine.head()

X = wine[['Proline', 'Total phenols', 'Hue', 'Nonflavanoid phenols']]
y = wine['Type']
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# SCore the model on the test data
print(knn.score(X_test, y_test))


# Log normalization
'''
 - Applies log transformation
 - Natural log using the constant  (2.718)
 - Captures relative changes, the magnitude of change, and keeps everything in the positive space

Checking the variance
 - Check the variance of the columns in the wine dataset.
'''
wine.describe()

'''
# Log normalization in Python
Now that we know that the Proline column in our wine dataset has a large amount of variance, let's log normalize it.
'''

# Print out the variance of the Proline column
print(wine['Proline'].var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])

# Check the variance of the normalized Proline column
print(wine['Proline_log'].var())


'''
# Scaling data for feature comparison
 - Features on different scales
 - Model with linear characteristics
 - Center features around 0 and transform to unit variance(1)
 - Transforms to approximately normal distribution

# Scaling data - investigating columns
We want to use the Ash, Alcalinity of ash, and Magnesium columns in the wine dataset to train a linear model, 
but it's possible that these columns are all measured in different ways, which would bias a linear model. 
Using describe() to return descriptive statistics about this dataset, which of the following statements are true 
about the scale of data in these columns?
'''
wine[['Ash', 'Alcalinity of ash', 'Magnesium']].describe()


# Scaling data - standardizing columns
'''
Since we know that the Ash, Alcalinity of ash, and Magnesium columns in the wine dataset are all on different scales, 
let's standardize them in a way that allows for use in a linear model.
'''
from sklearn.preprocessing import StandardScaler

# Create the scaler
ss = StandardScaler()

# Take a subset of the DataFrame you want to scale
wine_subset = wine[['Ash', 'Alcalinity of ash', 'Magnesium']]

print(wine_subset.iloc[:3])

# Apply the scaler to the DataFrame subset
wine_subset_scaled = ss.fit_transform(wine_subset)

print(wine_subset_scaled[:3])


# Standardized data and modeling
'''
# KNN on non-scaled data
Let's first take a look at the accuracy of a K-nearest neighbors model on the wine dataset without standardizing the data. 
The knn model as well as the X and y data and labels sets have been created already. 
Most of this process of creating models in scikit-learn should look familiar to you.
'''
wine = pd.read_csv('./dataset/wine_types.csv')

X = wine.drop('Type', axis=1)
y = wine['Type']

knn = KNeighborsClassifier()
# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))


'''
# KNN on scaled data
The accuracy score on the unscaled wine dataset was decent, but we can likely do better if we scale the dataset. 
The process is mostly the same as the previous exercise, with the added step of scaling the data.
'''
knn = KNeighborsClassifier()

# Create the scaling method
ss = StandardScaler()

# Apply the scaling method to the dataset used for modeling
X_scaled = ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Fit the k-nearest neighbors model to the training data.
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))