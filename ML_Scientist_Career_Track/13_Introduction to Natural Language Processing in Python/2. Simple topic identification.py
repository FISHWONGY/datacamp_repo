from pprint import pprint

# Word counts with bag-of-words
'''
Bag-of-words
 - Basic method for finding topics in a text
 - Need to first create tokens using tokenization
 - ... and then count up all the tokens
 - The more frequent a word, the more important it might be
 - Can be a great way to determine the significant words in a text
'''

# Bag-of-words picker
from nltk.tokenize import word_tokenize
from collections import Counter

my_string = "The cat is in the box. The cat box."
Counter(word_tokenize(my_string)).most_common(len(word_tokenize(my_string)))


# Building a Counter with bag-of-words
with open('/Online course/datacamp/13_Introduction to Natural Language Processing in Python/data/'
          'Wikipedia articles/wiki_text_debugging.txt', 'r') as file:
    article = file.read()
    article_title = word_tokenize(article)[2]

# Tokenize the aricle: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
pprint(bow_simple.most_common(10))


# Simple text preprocessing
'''
Preprocessing
 - Helps make for better input data
   - When performing machine learning or other statistical methods
Examples
 - Tokenization to create a bag of words
 - Lowercasing words
Lemmatization / Stemming
 - Shorten words to their root stems
Removing stop words, punctuation, or unwanted tokens
'''

import nltk
# nltk.download('wordnet')
with open('/Online course/datacamp/13_Introduction to Natural Language Processing in Python/data/'
          'english_stopwords.txt', 'r') as file:
    english_stops = file.read()

from nltk.stem import WordNetLemmatizer

# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]
print(alpha_only)

# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in english_stops]
print(no_stops)

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
print(lemmatized)

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
pprint(bow.most_common(10))


# Introduction to gensim
'''
Gensim
 - Popular open-source NLP library
 - Uses top academic models to perform complex tasks
   - Building document or word vectors
   - Performing topic identification and document comparison
'''

import glob

path_list = glob.glob(
    '/Online course/datacamp/13_Introduction to Natural Language Processing in Python/data/Wikipedia articles/*.txt')
articles = []
for article_path in path_list:
    article = []
    with open(article_path, 'r') as file:
        a = file.read()
    tokens = word_tokenize(a)
    lower_tokens = [t.lower() for t in tokens]

    # Retain alphabetic words: alpha_only
    alpha_only = [t for t in lower_tokens if t.isalpha()]

    # Remove all stop words: no_stops
    no_stops = [t for t in alpha_only if t not in english_stops]
    articles.append(no_stops)


from gensim.corpora.dictionary import Dictionary

# Create a Dictionary from the articles: dictionary
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get("computer")

# Use computer_id with the dictionary to print the word
print(dictionary.get(computer_id))

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])


# Gensim bag-of-words
from collections import defaultdict
import itertools

# Save the fifth document: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
    print(dictionary.get(word_id), word_count)

# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

# Create a sorted list from the defaultdict: sorted_word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)

# Print the top 5 words across all documents alongside the count
for word_id, word_count in sorted_word_count[:5]:
    print(dictionary.get(word_id), word_count)


# Tf-idf with gensim
'''
TF-IDF
 - Term Frequency - Inverse Document Frequency
 - Allows you to determine the most important words in each document
 - Each corpus may have shared words beyond just stop words
 - These words should be down-weighted in importance
 - Ensures most common words don't show up as key words
 - Keeps document specific frequent words wieghted high

'''

from gensim.models.tfidfmodel import TfidfModel

# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Print the first five weights
print(tfidf_weights[:5])

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)

