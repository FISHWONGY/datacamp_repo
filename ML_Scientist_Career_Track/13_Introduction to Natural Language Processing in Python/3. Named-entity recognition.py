from pprint import pprint
import matplotlib.pyplot as plt

# Named Entity Recognition
'''
Named Entity Recognition (NER)
 - NLP task to identify important named entities in the text
   - People, places, organizations
   - Dates, states, works of art
 - Can be used alongside topic identification
'''

# NER with NLTK
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

with open('/Online course/datacamp/13_Introduction to Natural Language Processing in Python/data/'
          'News articles/uber_apple.txt', 'r') as file:
    article = file.read()

from nltk.tokenize import sent_tokenize, word_tokenize

# Tokenize the article into sentences: sentences
sentences = sent_tokenize(article)
print(sentences)

# Tokenize each sentence into words: token_sentences
token_sentences = [word_tokenize(sent) for sent in sentences]
print(token_sentences)

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]
print(pos_sentences)

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == 'NE':
            print(chunk)


# Charting practice
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=False)

from collections import defaultdict

# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1

# Create a list from the dictionary keys for the cart labels: labels
labels = list(ner_categories.keys())
print(labels)

# Create a list of the values: values
# values = [ner_categories.get(v) for v in labels]
values = [ner_categories.get(l) for l in labels]
print(values)

# Create the pie chart
fig = plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)


# Introduction to SpaCy
'''
SpaCy
  - NLP library similar to gensim, with different implementations
  - Focuson creating NLP pipelines to generate models and corpora
  - Open source, with extra libraries and tools
     - Displacy

Why use SpaCy for NER?
  - Easy pipeline creation
  - Different entity types compared to nltk
  - Informal language corpora
    - Easily find entities in Tweets and chat messages
'''

# Comparing NLTK with spaCy NER
import spacy

'''
Terminal run - 
pip3 install -U pip setuptools wheel
pip3 install -U spacy
python3 -m spacy download en_core_web_sm
'''

# Instantiate the English model: nlp
nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load('en_core_web_sm', tagger=False, parser=False, matcher=False)

# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)


# Multilingual NER with polyglot
'''
polyglot
  - NLP library which uses word vectors
  - Vectors for many different languages (more than 130)
'''

# French NER with polyglot I
from polyglot.text import Text
# !polyglot download ner2.fr
# !polyglot download embeddings2.fr

with open(
        '/Online course/datacamp/13_Introduction to Natural Language Processing in Python/data/News articles/french.txt', 'r') as file:
    article = file.read()

print(article)

# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

txt.entities

# Print each of the entities found
for ent in txt.entities:
    print(ent)

# Print the type of ent
print(type(ent))


# French NER with polyglot II
# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

# Print entities
pprint(entities)


# Spanish NER with polyglot
# !polyglot download ner2.es embeddings2.es
with open(
        '/Online course/datacamp/13_Introduction to Natural Language Processing in Python/data/News articles/spanish.txt', 'r') as file:
    article = file.read()

txt = Text(article)

# Initialize the count variable: count
count = 0

# Iterate over all the entities
for ent in txt.entities:
    # check whether the entity contains 'Márquez' or 'Gabo'
    if ('Márquez' in ent) or ('Gabo' in ent):
        # Increment count
        count += 1

# Print count
print(count)

# Calculate the percentage of entities that refer to "Gabo": percentage
percentage = count / len(txt.entities)
print(percentage)

