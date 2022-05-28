from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# https://github.com/goodboychan/goodboychan.github.io/tree/master/_notebooks

iris = datasets.load_iris()
type(iris)
type(iris.data), type(iris.target)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])
X_new = np.array([[5.6, 2.8, 3.9, 1.1],
                  [5.7, 2.6, 3.8, 1.3],
                  [4.7, 3.2, 1.3, 0.2]])
prediction = knn.predict(X_new)
# X_new.shape
print('Prediction: {}'.format(prediction))

# Exercise k-Nearest Neighbors: Predict
df = pd.read_csv('/Online course/datacamp/Supervised Learning with scikit-learn/data/house-votes-84.csv')
df.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
              'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
              'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df.replace({'?': 'n'}, inplace=True)
df.replace({'n': 0, 'y':  1}, inplace=True)

# Create arrays for the features and the response variable
# y = get an np.array of the col I want to classify
y = df['party'].values
# X = rest of the col(features) that we use to predict the outcome - party in this case
X = df.drop('party', axis=1).values

# Classifier of neighbor = 6
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
# fit(features, outcome)
knn.fit(X, y)
# Predict the labels for the training data X
y_pred = knn.predict(X)

# Make up a sample to test the result (I think we will use test set normally?) 16 features as same as the model
# transpose = long to wide
X_new = pd.DataFrame([0.696469, 0.286139, 0.226851, 0.551315, 0.719469, 0.423106, 0.980764,
                      0.68483, 0.480932, 0.392118, 0.343178, 0.72905, 0.438572, 0.059678,
                      0.398044, 0.737995]).transpose()
# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

# My ownn practice - if there's 2 dataset
# if more than 1 testing data, need to use np.array instead of
X_new = np.array([[0.696469, 0.286139, 0.226851, 0.551315, 0.719469, 0.423106, 0.980764,
                   0.68483, 0.480932, 0.392118, 0.343178, 0.72905, 0.438572, 0.059678,
                   0.398044, 0.737995],
                  [0.396469, 0.986139, 0.236851, 0.151315, 0.919469, 0.413106, 0.180764,
                   0.28483, 0.480932, 0.792118, 0.353178, 0.12905, 0.938572, 0.009678,
                   0.098044, 0.237995]])
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))


# Measuring model performance
df = pd.read_csv('/Online course/datacamp/Supervised Learning with scikit-learn/data/house-votes-84.csv')
df.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
              'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
              'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df.replace({'?': 'n'}, inplace=True)
df.replace({'n': 0, 'y':  1}, inplace=True)

y = df['party'].values
X = df.drop('party', axis=1).values
###########################################
# Split the df into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=8)
# why 8? lol

# Build a model with the training data set
knn.fit(X_train, y_train)
# Store the result of the training data as y_pred
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

# Compare y_pred with y_test for the result/ confusion matrix
# user this knn.score thing to find the sweet spot for number of neighbours
knn.score(X_test, y_test)

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


# Exercise for measuring performance
# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits['DESCR'])

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')

