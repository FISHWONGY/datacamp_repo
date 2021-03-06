import pandas as pd
import numpy as np
from pprint import pprint

# Classifying fake news using supervised learning with NLP
'''
Supervised learning with NLP
  - Need to use language instead of geometric features
  - Use bag-of-words models or tf-idf features

Supervised learning steps
  - Collect and preprocess our data
  - Determine a label
  - Split data into training and test sets
  - Extract features from the text to help predict the label
  - Evaluate trained model using test set
'''

# Building word count vectors with scikit-learn
# CountVectorizer for text classification
df = pd.read_csv(
    '/Online course/datacamp/13_Introduction to Natural Language Processing in Python/data/fake_or_real_news.csv')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Create a series to store the labels: y
y = df.label

# Create training set and test set
X_train, X_test, y_train, y_test = train_test_split(df['text'], y,
                                                    test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'text' column values: count_train
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])


# TfidfVectorizer for text classification
'''
Similar to the sparse CountVectorizer created in the previous exercise, 
you'll work on creating tf-idf vectors for your documents. 
You'll set up a TfidfVectorizer and investigate some of its features.
'''
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# transform the test data: tfidf_test
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])


# Inspecting the vectors
# To get a better idea of how the vectors work, you'll investigate them by converting them into pandas DataFrames.

# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrame are equal
print(count_df.equals(tfidf_df))


# Training and testing a classification model with scikit-learn
# Training and testing the "fake news" model with CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
#        Actual varialbes (X), Actual outcome/ dependent variable (y)
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
# y_from_MLmodel= nb_classifier.predict(x test)
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
# Actual outcome/ dependent variable (y), pred = y that predicted from (x test set)
score = accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)


# Training and testing the "fake news" model with TfidfVectorizer
# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)


# Simple NLP, complex problems
# Improving your model
# Create the list of alphas: alphas
alphas = np.arange(0, 1, 0.1)


# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)

    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)

    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)

    # Compute accuracy: score
    score = accuracy_score(y_test, pred)
    return score


# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()


# Inspecting your model
# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array
# and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])

