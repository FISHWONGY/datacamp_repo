import pandas as pd
movies = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/'
                     'data/IMDB_sample.csv')

# Import the required vectorizer package and stop words list
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Define the vectorizer and specify the arguments
my_pattern = r'\b[^\d\W][^\d\W]+\b'
vect = TfidfVectorizer(ngram_range=(1, 2), max_features=100, token_pattern=my_pattern,
                       stop_words=ENGLISH_STOP_WORDS).fit(movies.review)

# Transform the vectorizer
X_txt = vect.transform(movies.review)

# Transform to a data frame and specify the column names
movies_df = pd.DataFrame(X_txt.toarray(), columns=vect.get_feature_names())
labels = movies["label"]
movies_df.insert(0, 'label', labels)

# Logistic regression of movie reviews
# Import the logistic regression
from sklearn.linear_model import LogisticRegression

# Define the vector of targets and matrix of features
y = movies_df.label
X = movies_df.drop(['label'], axis=1)

# Build a logistic regression model and calculate the accuracy
log_reg = LogisticRegression().fit(X, y)
print('Accuracy of logistic regression: ', log_reg.score(X, y))


# Logistic regression using Twitter data
# multi-class classification -
# airline_sentiment, which is 0 for negative tweets, 1 for neutral, and 2 for positive
tweets = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/data'
                     '/Tweets.csv')

tweets.loc[(tweets.airline_sentiment == 'negative'), 'airline_sentiment'] = 0
tweets.loc[(tweets.airline_sentiment == 'neutral'), 'airline_sentiment'] = 1
tweets.loc[(tweets.airline_sentiment == 'positive'), 'airline_sentiment'] = 2

tweets2 = tweets[['airline_sentiment', 'airline_sentiment_confidence', 'retweet_count']]


vect = TfidfVectorizer(ngram_range=(1, 2), max_features=100, token_pattern=my_pattern,
                       stop_words=ENGLISH_STOP_WORDS).fit(tweets.text)

# Transform the vectorizer
X_txt = vect.transform(movies.review)

# Transform to a data frame and specify the column names
tweets_df = pd.DataFrame(X_txt.toarray(), columns=vect.get_feature_names())
tweets2 = tweets2.join(tweets_df)

# Define the vector of targets and matrix of features
y = tweets2.airline_sentiment
X = tweets2.drop('airline_sentiment', axis=1)

from sklearn.metrics import accuracy_score, confusion_matrix
# Build a logistic regression model and calculate the accuracy
log_reg = LogisticRegression().fit(X, y)
print('Accuracy of logistic regression: ', log_reg.score(X, y))

# Create an array of prediction
y_predict = log_reg.predict(X)

# Print the accuracy using accuracy score
print('Accuracy of logistic regression: ', accuracy_score(y, y_predict))


# Build and assess a model: movies reviews
# Import the required packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Define the vector of labels and matrix of features
y = movies_df.label
X = movies_df.drop('label', axis=1)

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a logistic regression model and print out the accuracy
log_reg = LogisticRegression().fit(X_train, y_train)
print('Accuracy on train set: ', log_reg.score(X_train, y_train))
print('Accuracy on test set: ', log_reg.score(X_test, y_test))


# Performance metrics of Twitter data
tweets2 = tweets2.head(7500)
y = tweets2.airline_sentiment
X = tweets2.drop('airline_sentiment', axis=1)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y.astype('float64'),
                                                    test_size=0.3, random_state=123, stratify=y)

# Train a logistic regression
# log_reg = LogisticRegression().fit(X_train, y_train)
log_reg = LogisticRegression(solver='liblinear').fit(X_train, y_train)

# Make predictions on the test set
y_predicted = log_reg.predict(X_test)

# Print the performance metrics
print('Accuracy score test set: ', accuracy_score(y_test, y_predicted))
print('Confusion matrix test set: \n', confusion_matrix(y_test, y_predicted) / len(y_test))


# Build and assess a model: product reviews data
y = movies_df.label
X = movies_df.drop('label', axis=1)
# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a logistic regression
log_reg = LogisticRegression().fit(X_train, y_train)

# Predict the labels
y_predict = log_reg.predict(X_test)

# Print the performance metrics
print('Accuracy score of test data: ', accuracy_score(y_test, y_predict))
print('Confusion matrix of test data: \n', confusion_matrix(y_test, y_predict)/len(y_test))


# Predict probabilities of movie reviews
# Some pre-processing for the data
vect = TfidfVectorizer(ngram_range=(1, 2), max_features=200, token_pattern=my_pattern,
                       stop_words=ENGLISH_STOP_WORDS).fit(movies.review)

# Transform the vectorizer
X_txt = vect.transform(movies.review)

# Transform to a data frame and specify the column names
movies_df = pd.DataFrame(X_txt.toarray(), columns=vect.get_feature_names())
labels = movies["label"]
movies_df.insert(0, 'label', labels)

y = movies_df.label
X = movies_df.drop('label', axis=1)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)

# Train a logistic regression
log_reg = LogisticRegression().fit(X_train, y_train)

# Predict the probability of the 0 class
prob_0 = log_reg.predict_proba(X_test)[:, 0]
# Predict the probability of the 1 class
prob_1 = log_reg.predict_proba(X_test)[:, 1]

print("First 10 predicted probabilities of class 0: ", prob_0[:10])
print("First 10 predicted probabilities of class 1: ", prob_1[:10])


# Product reviews with regularization
tweets = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/data'
                     '/Tweets.csv')
tweets.loc[(tweets.airline_sentiment != 'positive'), 'airline_sentiment'] = 0
tweets.loc[(tweets.airline_sentiment == 'positive'), 'airline_sentiment'] = 1

vect = TfidfVectorizer(ngram_range=(1, 2), max_features=200, token_pattern=my_pattern,
                       stop_words=ENGLISH_STOP_WORDS).fit(tweets.text)

# Transform the vectorizer
X_txt = vect.transform(movies.review)

# Transform to a data frame and specify the column names
tweets_df = pd.DataFrame(X_txt.toarray(), columns=vect.get_feature_names())
airline_sentiments = tweets["airline_sentiment"]
tweets_df.insert(0, 'airline_sentiment', airline_sentiments)

y = tweets_df.airline_sentiment
X = tweets_df.drop('airline_sentiment', axis=1)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y.astype('float64'), test_size=0.2, random_state=123)

# Train a logistic regression with regularization of 1000
log_reg1 = LogisticRegression(solver='liblinear', C=1000).fit(X_train, y_train)
# Train a logistic regression with regularization of 0.001
log_reg2 = LogisticRegression(solver='liblinear', C=0.001).fit(X_train, y_train)

# Print the accuracies
print('Accuracy of model 1: ', log_reg1.score(X_test, y_test))
print('Accuracy of model 2: ', log_reg2.score(X_test, y_test))


# Regularizing models with Twitter data
# Build a logistic regression with regularizarion parameter of 100
log_reg1 = LogisticRegression(solver='liblinear', C=100).fit(X_train, y_train)
# Build a logistic regression with regularizarion parameter of 0.1
log_reg2 = LogisticRegression(solver='liblinear', C=0.1).fit(X_train, y_train)

# Predict the labels for each model
y_predict1 = log_reg1.predict(X_test)
y_predict2 = log_reg2.predict(X_test)

# Print performance metrics for each model
print('Accuracy of model 1: ', accuracy_score(y_test, y_predict1))
print('Accuracy of model 2: ', accuracy_score(y_test, y_predict2))
print('Confusion matrix of model 1: \n', confusion_matrix(y_test, y_predict1)/len(y_test))
print('Confusion matrix of model 2: \n', confusion_matrix(y_test, y_predict2)/len(y_test))


# Step 1: Word cloud and feature creation
import matplotlib.pyplot as plt
positive_review = movies[movies.label == 1]
positive_review = positive_review.head(100)
positive_reviews = positive_review['review'].str.cat(sep=' ')

# Create and generate a word cloud image
from wordcloud import WordCloud
cloud_positives = WordCloud(background_color='white').generate(positive_reviews)

# Display the generated wordcloud image
plt.imshow(cloud_positives, interpolation='bilinear')
plt.axis("off")

# Don't forget to show the final image
plt.show()


# Step 1: Word cloud and feature creation
reviews = movies[['label', 'review']]
# Tokenize each item in the review column
from nltk.tokenize import sent_tokenize, word_tokenize
word_tokens = [word_tokenize(review) for review in reviews.review]

# Create an empty list to store the length of the reviews
len_tokens = []

# Iterate over the word_tokens list and determine the length of each item
for i in range(len(word_tokens)):
     len_tokens.append(len(word_tokens[i]))

# Create a new feature for the lengh of each review
reviews['n_words'] = len_tokens


# Step 2: Building a vectorizer
# Import the TfidfVectorizer and default list of English stop words
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Build the vectorizer
vect = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), max_features=200, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(reviews.review)
# Create sparse matrix from the vectorizer
X = vect.transform(reviews.review)

# Create a DataFrame
reviews_transformed = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
print('Top 5 rows of the DataFrame: \n', reviews_transformed.head())


# Step 3: Building a classifier
# Define X and y
y = reviews_transformed.score
X = reviews_transformed.drop('score', axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=456)

# Train a logistic regression
log_reg = LogisticRegression().fit(X_train, y_train)
# Predict the labels
y_predicted = log_reg.predict(X_test)

# Print accuracy score and confusion matrix on test set
print('Accuracy on the test set: ', accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted)/len(y_test))