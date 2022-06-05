import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 8)

# Tokenization and Lemmatization
'''
Text preprocessing techniques
 - Converting words into lowercase
 - Removing leading and trailing whitespaces
 - Removing punctuation
 - Removing stopwords
 - Expanding contractions

Tokenization
 - the process of splitting a string into its constituent tokens

Lemmatization
 - the process of converting a word into its lowercased base form or lemma
'''

# Tokenizing the Gettysburg Address
with open(
        '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
        '14_Feature Engineering for NLP in Python/data/gettysburg.txt', 'r') as f:
    gettysburg = f.read()

import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# create a Doc object
doc = nlp(gettysburg)

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)


# Lemmatizing the Gettysburg address
# Print the gettysburg address
print(gettysburg)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

# Convert lemmas into a string
print(' '.join(lemmas))


#  Text cleaning
'''
Text cleaning techniques
 - Unnecessary whitespaces and escape sequences
 - Punctuations
 - Special characters (numbers, emojis, etc.)
 - Stopwords

Stopwords
 - Words that occur extremely commonly
 - E.g. articles, be verbs, pronouns, etc..
'''

# Cleaning a blog post
with open('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
          '14_Feature Engineering for NLP in Python/data/blog.txt', 'r') as file:
    blog = file.read()

stopwords = spacy.lang.en.stop_words.STOP_WORDS
blog = blog.lower()

# Generate doc Object: doc
doc = nlp(blog)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas))


# Cleaning TED talks in a dataframe
ted = pd.read_csv(
    '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
    '14_Feature Engineering for NLP in Python/data/ted.csv')
ted['transcript'] = ted['transcript'].str.lower()


# Function to preprocess text
def preprocess(text):
    # Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])

    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]

    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in stopwords]

    return ' '.join(a_lemmas)


# Apply preprocess to ted['transcript']
ted['transcript2'] = ted['transcript'].apply(preprocess)
print(ted['transcript2'])


# Part-of-speech tagging
'''
Part-of-Speech (POS)
 - helps in identifying distinction by identifying one bear as a noun and the other as a verb
 - Word-sense disambiguation
   - "The bear is a majestic animal"
   - "Please bear with me"
 - Sentiment analysis
 - Question answering
 - Fake news and opinion spam detection

- POS tagging
 - Assigning every word, its corresponding part of speech

POS annotation in spaCy
 - PROPN - proper noun
 - DET - determinant
'''

# POS tagging in Lord of the Flies
with open('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
          '14_Feature Engineering for NLP in Python/data/lotf.txt', 'r') as file:
    lotf = file.read()

# Create d Doc object
doc = nlp(lotf)

# Generate tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print(pos)

# Counting nouns in a piece of text
# Returns number of proper nouns
def proper_nouns(text, model=nlp):
    # Create doc object
    doc = model(text)

    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of proper nouns
    return pos.count('PROPN')


print(proper_nouns('Abdul, Bill and Cathy went to the market to buy apples.', nlp))


# Returns number of other nouns
def nouns(text, model=nlp):
    # create doc object
    doc = model(text)

    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of other nouns
    return pos.count('NOUN')


print(nouns('Abdul, Bill and Cathy went to the market to buy apples.', nlp))


# Noun usage in fake news
headlines = pd.read_csv(
    '/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
    '14_Feature Engineering for NLP in Python/data/fakenews.csv')

headlines['num_propn'] = headlines['title'].apply(proper_nouns)
headlines['num_noun'] = headlines['title'].apply(nouns)

# Compute mean of proper nouns
real_propn = headlines[headlines['label'] == 'REAL']['num_propn'].mean()
fake_propn = headlines[headlines['label'] == 'FAKE']['num_propn'].mean()

# Compute mean of other nouns
real_noun = headlines[headlines['label'] == 'REAL']['num_noun'].mean()
fake_noun = headlines[headlines['label'] == 'FAKE']['num_noun'].mean()

# Print results
print("Mean no. of proper nouns in real and fake headlines are %.2f and %.2f respectively" %
      (real_propn, fake_propn))
print("Mean no. of other nouns in real and fake headlines are %.2f and %.2f respectively" %
     (real_noun, fake_noun))


# Named entity recognition
'''
Named entity recognition (NER)
 - Identifying and classifying named entities into predefined categories
 - Categories include person, organization, country, etc.
'''

# Create a Doc instance
text = 'Sundar Pichai is the CEO of Google. Its headquarter is in Mountain View.'
doc = nlp(text)

# Print all named entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)


# Identifying people mentioned in a news article
with open('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
          '14_Feature Engineering for NLP in Python/data/tc.txt', 'r') as file:
    tc = file.read()


def find_persons(text):
    # Create Doc object
    doc = nlp(text)

    # Indentify the persons
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

    # Return persons
    return persons


print(find_persons(tc))

