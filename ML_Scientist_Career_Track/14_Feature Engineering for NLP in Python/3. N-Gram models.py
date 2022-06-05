import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy

plt.rcParams['figure.figsize'] = (8, 8)

# Building a bag of words model
'''
Bag of words model
 - Extract word tokens
 - Compute frequency of word tokens
 - Construct a word vector out of these frequencies and vocabulary of corpus
'''

# BoW model for movie taglines
movies = pd.read_csv(
    '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
    '14_Feature Engineering for NLP in Python/data/movie_overviews.csv').dropna()
movies['tagline'] = movies['tagline'].str.lower()

corpus = movies['tagline']

from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Print the shape of bow_matrix
print(bow_matrix.shape)


# Analyzing dimensionality and preprocessing
nlp = spacy.load('en_core_web_sm')
stopwords = spacy.lang.en.stop_words.STOP_WORDS

lem_corpus = corpus.apply(lambda row: ' '.join([t.lemma_ for t in nlp(row)
                                                if t.lemma_ not in stopwords
                                                and t.lemma_.isalpha()]))

print(lem_corpus)

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate of word vectors
bow_lem_matrix = vectorizer.fit_transform(lem_corpus)

# Print the shape of how_lem_matrix
print(bow_lem_matrix.shape)


# Mapping feature indices with feature names
sentences = ['The lion is the king of the jungle',
             'Lions have lifespans of a decade',
             'The lion is an endangered species']

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(sentences)

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary
bow_df.columns = vectorizer.get_feature_names()

# Print bow_df
bow_df


# Building a BoW Naive Bayes classifier
'''
Steps
 - Text preprocessing
 - Building a bag-of-words model (or representation)
 - Machine Learning
'''

# BoW vectors for movie reviews
# Sentiment - 0 = negative, 1 = positive
movie_reviews = pd.read_csv(
    '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
    '14_Feature Engineering for NLP in Python/data/movie_reviews_clean.csv')

X = movie_reviews['review']
y = movie_reviews['sentiment']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create a CounterVectorizer object
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_bow = vectorizer.transform(X_test)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)


# Predicting the sentiment of a movie review
from sklearn.naive_bayes import MultinomialNB

# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the classifier
y_train[np.isnan(y_train)] = np.median(y_train[~np.isnan(y_train)])
clf.fit(X_train_bow, y_train)

# Measure the accuracy
y_test[np.isnan(y_test)] = np.median(y_test[~np.isnan(y_test)])
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = 'The movie was terrible. The music was underwhelming and the acting mediocre.'
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))


# Building n-gram models
'''
BoW shortcomings
 - Example
   - The movie was good and not boring -> positive
   - The movie was not good and boring -> negative
 - Exactly the same BoW representation!
 - Context of the words is lost.
 - Sentiment dependent on the position of not

n-grams
 - Contiguous sequence of n elements (or words) in a given document.
 - Bi-grams / Tri-grams

n-grams Shortcomings
 - Increase number of dimension, occurs curse of dimensionality
 - Higher order n-grams are rare
'''

# n-gram models for movie tag lines
# Generate n-grams upto n=1
vectorizer_ng1 = CountVectorizer(ngram_range=(1, 1))
ng1 = vectorizer_ng1.fit_transform(corpus)

# Generate n-grams upto n=2
vectorizer_ng2 = CountVectorizer(ngram_range=(1, 2))
ng2 = vectorizer_ng2.fit_transform(corpus)

# Generate n-grams upto n=3
vectorizer_ng3 = CountVectorizer(ngram_range=(1, 3))
ng3 = vectorizer_ng3.fit_transform(corpus)

# Print the number of features for each model
print("ng1, ng2 and ng3 have %i, %i and %i features respectively" %
      (ng1.shape[1], ng2.shape[1], ng3.shape[1]))


# Higher order n-grams for sentiment analysis
ng_vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_ng = ng_vectorizer.fit_transform(X_train)
X_test_ng = ng_vectorizer.transform(X_test)

# Define an instance of MultinomialNB
clf_ng = MultinomialNB()

# Fit the classifier
clf_ng.fit(X_train_ng, y_train)

# Measure the accuracy
accuracy = clf_ng.score(X_test_ng, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = 'The movie was not good. The plot had several holes and the acting lacked panache'
prediction = clf_ng.predict(ng_vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))

# Comparing performance of n-gram models
y_train[np.isnan(y_train)] = np.median(y_train[~np.isnan(y_train)])
y_test[np.isnan(y_test)] = np.median(y_test[~np.isnan(y_test)])

import time

# Splitting the data into training and test sets
# stratify gives an error if the below line is not added
movie_reviews['sentiment'].fillna(method='ffill', inplace=True)

start_time = time.time()
train_X, test_X, train_y, test_y = train_test_split(movie_reviews['review'],
                                                    movie_reviews['sentiment'],
                                                    test_size=0.5,
                                                    random_state=42,
                                                    stratify=movie_reviews['sentiment'])
# Generateing ngrams
vectorizer = CountVectorizer(ngram_range=(1, 1))
train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

# Fit classifier
clf = MultinomialNB()
clf.fit(train_X, train_y)

# Print the accuracy, time and number of dimensions
print("The program took %.3f seconds to complete. The accuracy on the test set is %.2f. " %
      (time.time() - start_time, clf.score(test_X, test_y)))
print("The ngram representation had %i features." % (train_X.shape[1]))


#######
start_time = time.time()

# Splitting the data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(movie_reviews['review'],
                                                    movie_reviews['sentiment'],
                                                    test_size=0.5,
                                                    random_state=42,
                                                    stratify=movie_reviews['sentiment'])

# Generateing ngrams
vectorizer = CountVectorizer(ngram_range=(1, 3))
train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

# Fit classifier
clf = MultinomialNB()
clf.fit(train_X, train_y)

# Print the accuracy, time and number of dimensions
print("The program took %.3f seconds to complete. The accuracy on the test set is %.2f. " %
      (time.time() - start_time, clf.score(test_X, test_y)))
print("The ngram representation had %i features." % (train_X.shape[1]))

