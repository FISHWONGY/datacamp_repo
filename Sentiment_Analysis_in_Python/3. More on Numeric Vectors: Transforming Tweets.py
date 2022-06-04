import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Word cloud of tweets
tweets = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/data'
                    '/Tweets.csv')
text_tweet = tweets['text'].str.cat(sep=' ')


# Create and generate a word cloud image
my_cloud = WordCloud(background_color='white').generate(text_tweet)

# Display the generated wordcloud image
plt.imshow(my_cloud, interpolation='bilinear')
plt.axis("off")

# Don't forget to show the final image
plt.show()


# Define the default list of stop words and update it.
# Import the word cloud function and stop words list
from wordcloud import STOPWORDS

# Define the list of stopwords
my_stop_words = STOPWORDS.update(['airline', 'airplane'])

# Create and generate a word cloud image
my_cloud = WordCloud(stopwords=my_stop_words).generate(text_tweet)

# Display the generated wordcloud image
plt.imshow(my_cloud, interpolation='bilinear')
plt.axis("off")
# Don't forget to show the final image
plt.show()


# Airline sentiment with stop words
# Import the stop words
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Define the stop words
my_stop_words = ENGLISH_STOP_WORDS.union(['airline', 'airlines', '@'])

# Build and fit the vectorizer
vect = CountVectorizer(stop_words=my_stop_words)
vect.fit(tweets.text)

# Create the bow representation
X_review = vect.transform(tweets.text)
# Create the data frame
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())
print(X_df.head())


# Create and generate a word cloud image
my_cloud = WordCloud(stopwords=my_stop_words).generate(text_tweet)
# Display the generated wordcloud image
plt.imshow(my_cloud, interpolation='bilinear')
plt.axis("off")
# Don't forget to show the final image
plt.show()


# Multiple text columns
# Define the stop words
my_stop_words = ENGLISH_STOP_WORDS.union(['airline', 'airlines', '@', 'am', 'pm'])

# Build and fit the vectorizers
vect1 = CountVectorizer(stop_words=my_stop_words)
vect2 = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
vect1.fit(tweets.text)
vect2.fit(tweets.negativereason.astype('U'))

# Print the last 15 features from the first, and all from second vectorizer
print(vect1.get_feature_names()[-15:])
print(vect2.get_feature_names())


# Specify the token pattern
# vectorize the object column using CountVectorizer
# Build and fit the vectorizer
vect = CountVectorizer(token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(tweets.text)
vect.transform(tweets.text)

print('Length of vectorizer: ', len(vect.get_feature_names()))


# Build the first vectorizer
vect1 = CountVectorizer().fit(tweets.text)
vect1.transform(tweets.text)


# Build a second vectorizer, specifying the pattern of tokens to be equal to r'\b[^\d\W][^\d\W]'
# Build the second vectorizer
vect2 = CountVectorizer(token_pattern=r'\b[^\d\W][^\d\W]').fit(tweets.text)
vect2.transform(tweets.text)

# Print out the length of each vectorizer
print('Length of vectorizer 1: ', len(vect1.get_feature_names()))
print('Length of vectorizer 2: ', len(vect2.get_feature_names()))


# String operators with the Twitter data
# Task - Turn the text column into a list of tokens. Then, using string operators,
# remove all non-alphabetic characters from the created list of tokens
# Import the word tokenizing package
from nltk import word_tokenize

# Tokenize the text column
word_tokens = [word_tokenize(review) for review in tweets.text]
print('Original tokens: ', word_tokens[0])

# Filter out non-letter characters
cleaned_tokens = [[word for word in item if word.isalpha()] for item in word_tokens]
print('Cleaned tokens: ', cleaned_tokens[0])


# More string operators and Twitter
'''
You need to construct three new lists by applying different string operators:

a list retaining only letters
a list retaining only characters
a list retaining only digits
'''

tweets_list = ["@VirginAmerica it's really aggressive to blast obnoxious 'entertainment' in your guests' faces &amp; "
               "they have little recourse",
               "@VirginAmerica Hey, first time flyer next week - excited! But I'm having a hard time getting my "
               "flights added to my Elevate account. Help?",
               '@united Change made in just over 3 hours. For something that should have taken seconds online, '
               'I am not thrilled. Loved the agent, though.']

# Create a list of lists, containing the tokens from list_tweets
tokens = [word_tokenize(item) for item in tweets_list]

# Remove characters and digits , i.e. retain only letters
letters = [[word for word in item if word.isalpha()] for item in tokens]
# Remove characters, i.e. retain only letters and digits
let_digits = [[word for word in item if word.isalnum()] for item in tokens]
# Remove letters and characters, retain only digits
digits = [[word for word in item if word.isdigit()] for item in tokens]

# Print the last item in each list
print('Last item in alphabetic list: ', letters[2])
print('Last item in list of alphanumerics: ', let_digits[2])
print('Last item in the list of digits: ', digits[2])


# Stems and lemmas from GoT
# Build a list of tokens from the Game of Throne string
GoT = "Never forget what you are, for surely the world will not. Make it your strength. " \
      "Then it can never be your weakness. Armour yourself in it, and it will never be used to hurt you."
# Import the required packages from nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize

porter = PorterStemmer()
WNlemmatizer = WordNetLemmatizer()

# Tokenize the GoT string
tokens = word_tokenize(GoT)


# Using list comprehension and the porter stemmer you imported, create the stemmed_tokens list.
import time

# Log the start time
start_time = time.time()

# Build a stemmed list
stemmed_tokens = [porter.stem(token) for token in tokens]

# Log the end time
end_time = time.time()

print('Time taken for stemming in seconds: ', end_time - start_time)
print('Stemmed tokens: ', stemmed_tokens)


# Using list comprehension and the WNlemmatizer you imported, create the lem_tokens list.
# Log the start time
start_time = time.time()

# Build a lemmatized list
lem_tokens = [WNlemmatizer.lemmatize(token) for token in tokens]

# Log the end time
end_time = time.time()

print('Time taken for lemmatizing in seconds: ', end_time - start_time)
print('Lemmatized tokens: ', lem_tokens)


# Stem Spanish reviews

reviews = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/data'
                      '/amazon_reviews_sample.csv')

non_en_index = [1249, 1259, 1260, 1261, 1639, 1745, 2316, 2486, 2760, 2903, 2908, 3318, 3694, 4820, 4914, 5720, 5875, 5901, 6234, 6631, 7078, 7307, 7888, 7983, 8018, 8340, 9265, 9422, 9624]
non_english_reviews = reviews[reviews['Unnamed: 0'].isin(non_en_index)]
non_english_reviews = non_english_reviews[["score", "review"]]
# Import the language detection package
import langdetect

# Loop over the rows of the dataset and append
languages = []
for i in range(len(non_english_reviews)):
    languages.append(langdetect.detect_langs(non_english_reviews.iloc[i, 1]))

# Clean the list by splitting
languages = [str(lang).split(':')[0][1:] for lang in languages]
# Assign the list to a new feature
non_english_reviews['language'] = languages

# Select the Spanish ones
filtered_reviews = non_english_reviews[non_english_reviews.language == 'es']


# create word tokens from the Spanish reviews and will stem them using a SnowBall stemmer for Spanish
# Import the required packages
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize

# Import the Spanish SnowballStemmer
SpanishStemmer = SnowballStemmer("spanish")

# Create a list of tokens
tokens = [word_tokenize(review) for review in filtered_reviews.review]
# Stem the list of tokens
stemmed_tokens = [[SpanishStemmer.stem(word) for word in token] for token in tokens]

# Print the first item of the stemmed tokenss
print(stemmed_tokens[0])


# Stems from tweets
# work with this array and transform it into a list of tokens using list comprehension.
tweets_list = tweets['text'].to_list()
# Call the stemmer
porter = PorterStemmer()

# Transform the array of tweets to tokens - this has to be a list
tokens = [word_tokenize(tweet) for tweet in tweets_list]
# Stem the list of tokens
stemmed_tokens = [[porter.stem(word) for word in tweet] for tweet in tokens]
# Print the first element of the list
print(stemmed_tokens[0])


# Your first TfIdf
# Import the required function
from sklearn.feature_extraction.text import TfidfVectorizer

annak = ['Happy families are all alike;', 'every unhappy family is unhappy in its own way']

# Call the vectorizer and fit it
anna_vect = TfidfVectorizer().fit(annak)

# Create the tfidf representation
anna_tfidf = anna_vect.transform(annak)

# Print the result
print(anna_tfidf.toarray())

# Import the required vectorizer package and stop words list
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Define the vectorizer and specify the arguments
my_pattern = r'\b[^\d\W][^\d\W]+\b'
vect = TfidfVectorizer(ngram_range=(1, 2), max_features=100, token_pattern=my_pattern,
                       stop_words=ENGLISH_STOP_WORDS).fit(tweets.text)

# Transform the vectorizer
X_txt = vect.transform(tweets.text)

# Transform to a data frame and specify the column names
X = pd.DataFrame(X_txt.toarray(), columns=vect.get_feature_names())
print('Top 5 rows of the DataFrame: ', X.head())


# Tfidf and a BOW(bag of Word) on same data
# transform the review column of the Amazon product reviews using both a bag-of-words and a tfidf transformation
# Import the required packages
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Build a BOW and tfidf vectorizers from the review column and with max of 100 features
vect1 = CountVectorizer(max_features=100).fit(reviews.review)
vect2 = TfidfVectorizer(max_features=100).fit(reviews.review)

# Transform the vectorizers
X1 = vect1.transform(reviews.review)
X2 = vect2.transform(reviews.review)
# Create DataFrames from the vectorizers
X_df1 = pd.DataFrame(X1.toarray(), columns=vect1.get_feature_names())
X_df2 = pd.DataFrame(X2.toarray(), columns=vect2.get_feature_names())
print('Top 5 rows, using BOW: \n', X_df1.head())
print('Top 5 rows using tfidf: \n', X_df2.head())