# Bags of words

# Import the required function
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

annak = ['Happy families are all alike;', 'every unhappy family is unhappy in its own way']

# Build the vectorizer and fit it
anna_vect = CountVectorizer()
anna_vect.fit(annak)

# Create the bow representation
anna_bow = anna_vect.transform(annak)

# Print the bag-of-words result
print(anna_bow.toarray())


# BOW using product reviews
reviews = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/data'
                      '/amazon_reviews_sample.csv')
# Build the vectorizer, specify max features
vect = CountVectorizer(max_features=100)
# Fit the vectorizer
vect.fit(reviews.review)

# Transform the review column
X_review = vect.transform(reviews.review)

# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())


# Specify token sequence length with BOW - ngram_range=(1, 2)
# Build the vectorizer, specify token sequence and fit
vect = CountVectorizer(ngram_range=(1, 2))
vect.fit(reviews.review)

# Transform the review column
X_review = vect.transform(reviews.review)

# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())


# Size of vocabulary of movies reviews
movies = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/'
                     'data/IMDB_sample.csv')
# Build the vectorizer, specify size of vocabulary and fit
# TRY limit the size of the vocabulary to include terms which occur in no more than 100 documents.
vect = CountVectorizer(max_features=100)
vect.fit(movies.review)

# Transform the review column
X_review = vect.transform(movies.review)
# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())

# TRY limit the size of the vocabulary to include terms which occur in no more than 200 documents.
# max_df = 200
# Build and fit the vectorizer
vect = CountVectorizer(max_df=200)
vect.fit(movies.review)

# Transform the review column
X_review = vect.transform(movies.review)
# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())


# Try limit the size of the vocabulary to ignore terms which occur in less than 50 documents
# Build and fit the vectorizer
vect = CountVectorizer(min_df=50)
vect.fit(movies.review)

# Transform the review column
X_review = vect.transform(movies.review)
# Create the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())


# BOW with n-grams and vocabulary size
# Build the vectorizer, specify max features and fit
# only want 1000 words and they have to appear at least 500 documents
vect = CountVectorizer(max_features=1000, ngram_range=(2, 2), max_df=500)
vect.fit(reviews.review)

# Transform the review
X_review = vect.transform(reviews.review)

# Create a DataFrame from the bow representation
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())


# Tokenize a string from GoT
with open(
        '/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/data'
        '/GoT.txt', 'r') as f:
    GoT = f.read()

# Import the required function
from nltk import word_tokenize

# Transform the GoT string to word tokens
print(word_tokenize(GoT))


avengers = ["Cause if we can't protect the Earth, you can be d*** sure we'll avenge it",
            'There was an idea to bring together a group of remarkable people, to see if we could become something more',
            "These guys come from legend, Captain. They're basically Gods."]

# Tokenize each item in the avengers
tokens_avengers = [word_tokenize(item) for item in avengers]

print(tokens_avengers)

# A feature for the length of a review
# Tokenize each item in the review column
word_tokens = [word_tokenize(review) for review in reviews.review]

# Print out the first item of the word_tokens list
print(word_tokens[0])

# Create a new feature for the length of a review, using the familiar reviews dataset.
# Create an empty list to store the length of the reviews
len_tokens = []

# Iterate over the word_tokens list and determine the length of each item
for i in range(len(word_tokens)):
     len_tokens.append(len(word_tokens[i]))

# Create a new feature for the lengh of each review
reviews['n_words'] = len_tokens


# Identify the language of a string
# Import the language detection function and package
from langdetect import detect_langs

foreign = "L'histoire rendu était fidèle, excellent, et grande."

# Detect the language of the foreign string
print(detect_langs(foreign))


# detect the language of each item in a list
sentences = ["L'histoire rendu était fidèle, excellent, et grande.",
             'Excelente muy recomendable.',
             'It had a leak from day one but the return and exchange process was very quick.']
languages = []

# Loop over the sentences in the list and detect their language
for sentence in sentences:
    languages.append(detect_langs(sentence))

print('The detected languages are: ', languages)


# Language detection of product reviews
non_en_index = [1249, 1259, 1260, 1261, 1639, 1745, 2316, 2486, 2760, 2903, 2908, 3318, 3694, 4820, 4914, 5720, 5875, 5901, 6234, 6631, 7078, 7307, 7888, 7983, 8018, 8340, 9265, 9422, 9624]
non_english_reviews = reviews[reviews['Unnamed: 0'].isin(non_en_index)]
non_english_reviews = non_english_reviews[["score", "review"]]

languages = []

# Loop over the rows of the dataset and append
for row in range(len(non_english_reviews)):
    languages.append(detect_langs(non_english_reviews.iloc[row, 1]))

# Clean the list by splitting
languages = [str(lang).split(':')[0][1:] for lang in languages]

# Assign the list to a new feature
non_english_reviews['language'] = languages

print(non_english_reviews.head())