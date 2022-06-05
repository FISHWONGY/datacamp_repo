import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 8)

# Introduction to NLP feature engineering
df1 = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                  '14_Feature Engineering for NLP in Python/data/FE_df1.csv')
# Print the features of df1
print(df1.columns)

# Perform one-hot encoding
df1 = pd.get_dummies(df1, columns=['feature 5'])

# Print the new features of df1
print(df1.columns)


# Basic feature extraction
# Character count of Russian tweets
tweets = pd.read_csv(
    '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
    '14_Feature Engineering for NLP in Python/data/russian_tweets.csv')

# Create a feature char_count
tweets['char_count'] = tweets['content'].astype('str').apply(len)


# Print the average character count
print(tweets['char_count'].mean())


# Word count of TED talks
ted = pd.read_csv(
    '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
    '14_Feature Engineering for NLP in Python/data/ted.csv')


# Function that returns number of words in a string
def count_words(string):
    # Split the string into words
    words = string.split()

    # Return the number of words
    return len(words)


# Create a new feature word_count
ted['word_count'] = ted['transcript'].apply(count_words)

# Print the average word count of the talks
print(ted['word_count'].mean())


# Hashtags and mentions in Russian tweets
# Function that returns number of hashtags in a string
def count_hashtags(string):
    # Split the string into words
    words = string.split()

    # Create a list of words that are hashtags
    hashtags = [word for word in words if word.startswith('#')]

    # Return number of hashtags
    return (len(hashtags))


tweets['word_count'] = tweets['content'].astype('str').apply(count_words)
# Create a feature hashtag_countand display distribution
tweets['hashtag_count'] = tweets['content'].astype('str').apply(count_hashtags)
tweets['hashtag_count'].hist()
plt.title('Hashtag count distribution')


# Function that returns number of mentions in a string
def count_mentions(string):
    # Split the string into words
    words = string.split()

    # Create a list of words that are mentions
    mentions = [word for word in words if word.startswith('@')]

    # Return number of mentions
    return (len(mentions))


# Create a feature mention_count and display distribution
tweets['mention_count'] = tweets['content'].astype('str').apply(count_mentions)
tweets['mention_count'].hist()
plt.title('Mention count distribution')


# Readability tests
'''
Readability test
Determine readability of an English passage
Scale ranging from primary school up to college graduate level
A mathematical formula utilizing word, syllable and sentence count
Used in fake news and opinion spam detection

Examples
 - Flesch reading ease
 - Gunning fog index
 - Simple Measure of Gobbledygook (SMOG)
 - Dale-Chall score

Flesch reading ease
 - One of the oldest and most widely used tests
 - Dependent on two factors
   - Greater the average sentence length, harder the text is to read
   - Greater the average number of syllables in a word, harder the text is to read
 - Higher the score, greater the readability

Gunning fog index
 - Developed in 1954
 - Also dependent on average sentence length
 - Greater the percentage of complex words, harder the text is to read
 - Higher the index, lesser the readability
'''

# Readability of 'The Myth of Sisyphus'
with open(
        '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
        '14_Feature Engineering for NLP in Python/data/sisyphus_essay.txt', 'r') as f:
    sisyphus_essay = f.read()

sisyphus_essay[:100]

from textatistic import Textatistic

# Compute the readability scores - needs to be a str
readability_scores = Textatistic(sisyphus_essay).scores

# Print the flesch reading ease score - most widely ised
flesch = readability_scores['flesch_score']
print('The Flesch Reading Ease is %.2f' % (flesch))


# Readability of various publications
'''
The excerpts are available as the following strings:

 - forbes- An excerpt from an article from Forbes magazine on the Chinese social credit score system.
 - harvard_law- An excerpt from a book review published in Harvard Law Review.
 - r_digest- An excerpt from a Reader's Digest article on flight turbulence.
 - time_kids - An excerpt from an article on the ill effects of salt consumption published in TIME for Kids.
'''

import glob
texts = []
text_list = glob.glob(
    '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
    '14_Feature Engineering for NLP in Python/data/exercise1/*.txt')

text_list

for text in text_list:
    if text != '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/' \
               '14_Feature Engineering for NLP in Python/data/sisyphus_essay.txt':
        with open(text, 'r') as f:
            texts.append(f.read())

time_kids, forbes, r_digest, harvard_law = texts

# List of excerpts
excerpts = [forbes, harvard_law, r_digest, time_kids]

# Loop through excerpts and compute gunning fog index
gunning_fog_scores = []
for excerpt in excerpts:
    readability_scores = Textatistic(excerpt).scores
    gunning_fog = readability_scores['gunningfog_score']
    gunning_fog_scores.append(gunning_fog)

# Print the gunning fog indices
print(gunning_fog_scores)

