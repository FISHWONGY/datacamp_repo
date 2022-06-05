import pandas as pd
import numpy as np

# Building tf-idf document vectors
'''
n-gram modeling
 - Weight of dimension dependent on the frequency of the word corresponding to the dimension

Applications
 - Automatically detect stopwords
 - Search
 - Recommender systems
 - Better performance in predictive modeling for some cases

Term frequency-inverse document frequency
 - Proportional to term frequency
 - Inverse function of the number of documents in which it occurs
'''

# tf-idf vectors for TED talks
df = pd.read_csv(
    '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
    '14_Feature Engineering for NLP in Python/data/ted.csv')

ted = df['transcript']

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)


# Cosine similarity
# Initialize numpy vectors
A = np.array([1, 3])
B = np.array([-2, 2])

# Compute dot product
dot_prod = np.dot(A, B)

# Print dot product
print(dot_prod)


# Cosine similarity matrix of a corpus
corpus = ['The sun is the largest celestial body in the solar system',
          'The solar system consists of the sun and eight revolving planets',
          'Ra was the Egyptian Sun God',
          'The Pyramids were the pinnacle of Egyptian architecture',
          'The quick brown fox jumps over the lazy dog']

from sklearn.metrics.pairwise import cosine_similarity

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)


# Building a plot line based recommender
'''
Steps
 - Text preprocessing
 - Generate tf-idf vectors
 - Generate cosine-similarity matrix

The recommender function
 - Take a movie title, cosine similarity matrix and indices series as arguments
 - Extract pairwise cosine similarity scores for the movie
 - Sort the scores in descending order
 - Output titles corresponding to the highest scores
 - Ignore the highest similarity score (of 1)
'''

import time

# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" % (time.time() - start))


from sklearn.metrics.pairwise import linear_kernel

# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" % (time.time() - start))


#  The recommender function
metadata = pd.read_csv(
    '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
    '14_Feature Engineering for NLP in Python/data/movie_metadata.csv').dropna()

# Generate mapping between titles and index
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


def get_recommendations(title, cosine_sim, indices):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]


# Plot recommendation engine
movie_plots = metadata['overview']

# Initialize the TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_plots)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Generate recommendations
print(get_recommendations("The Dark Knight Rises", cosine_sim, indices))


# TED talk recommender
ted = pd.read_csv(
    '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
    '14_Feature Engineering for NLP in Python/data/ted_clean.csv', index_col=0)

ted = ted.dropna()


def get_recommendations(title, cosine_sim, indices):
    # Get the index of the movie that matches the title
    idx = int(indices[title]) # IMPORTANT - Hv to change this to int to get sim_scores
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    talk_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return ted['title'].iloc[talk_indices]


# Generate mapping between titles and index
indices = pd.Series(ted.index, index=ted['title']).drop_duplicates()
transcripts = ted['transcript']

# Initialize the TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Generate recommendations
print(get_recommendations('5 ways to kill your dreams', cosine_sim, indices))


# Beyond n-grams: word embeddings
'''
Word embeddings
 - Mapping words into an n-dimensional vector space
 - Produced using deep learning and huge amounts of data
 - Discern how similar two words are to each other
 - Used to detect synonyms and antonyms
 - Captures complex relationships
 - Dependent on spacy model; independent of dataset you use
'''

import spacy
# python3 -m spacy download en_core_web_lg
nlp = spacy.load('en_core_web_lg')

# Generating word vectors
sent = 'I like apples and orange'

# Create the doc object
doc = nlp(sent)

# Compute pairwise similarity scores
for token1 in doc:
    for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))


# Computing similarity of Pink Floyd songs
with open('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
          '14_Feature Engineering for NLP in Python/data/mother.txt', 'r') as f:
    mother = f.read()

with open('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
          '14_Feature Engineering for NLP in Python/data/hopes.txt', 'r') as f:
    hopes = f.read()

with open('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
          '14_Feature Engineering for NLP in Python/data/hey.txt', 'r') as f:
    hey = f.read()

# Create Doc objects
mother_doc = nlp(mother)
hopes_doc = nlp(hopes)
hey_doc = nlp(hey)

# Print similarity between mother and hopes
print(mother_doc.similarity(hopes_doc))

# Print similarity between mother and hey
print(mother_doc.similarity(hey_doc))

