import re
from pprint import pprint

# Python re Module
'''
split: split a string on regex
findall: find all patterns in a string
search: search for a pattern
match: match an entire string or substring based on a pattern

Pattern first, and the string second
May return an iterator, string, or match object
'''

my_string = "Let's write RegEx!"
PATTERN = r"\w+"
re.findall(PATTERN, my_string)


# Practicing regular expressions - re.split() and re.findall()
my_string = "Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"

# Write a pattern to match sentence endings: sentence_endings
sentence_endings = r"[.?!]"

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings, my_string))

# Find all capicalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))

# Split my_string on spaces and print the result
spaces = r"\s+"
print(re.split(spaces, my_string))

# Find all digits in my_string and print the result
digits = r"\d+"
print(re.findall(digits, my_string))

# Introduction to tokenization
'''
Tokenization:
- Turning a string or document into tokens (smaller chunks)
- One step in preparing a text for NLP
- Many different theories and rules
- You can create your own rules using regular expressions
Some examples:
- Breaking out words or sentences
- Separating punctuation
- Separating all hashtags in a tweet

Why tokenize?
- Easier to map part of speech
- Matching common words
- Removing unwanted tokens

Other nltk tokenizers
- sent_tokenize: tokenize a document into sentences
- regexp_tokenize: tokenize a string or document based on a regular expression pattern
- TweetTokenizer: special class just for tweet tokenization, allowing you to separate hashtags, mentions and lots of exclamation points
'''

# Word tokenization with NLTK
with open('/Online course/datacamp/13_Introduction to Natural Language Processing in Python/data/grail.txt', 'r') as file:
    holy_grail = file.read()
    scene_one = re.split('SCENE 2:', holy_grail)[0]

print(scene_one)


from nltk.tokenize import word_tokenize, sent_tokenize

# Split scene_one into sentences: sentences
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens)


# More regex with re.search()
# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search("coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*\]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))


# Advanced tokenization with NLTK and regex
'''
Regex groups using or |
 - OR is represented using |
 - You can define a group using ()
 - You can define explicit character ranges using []
'''

# Choosing a tokenizer
from nltk.tokenize import regexp_tokenize

my_string = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"

pattern1 = r'(\\w+|\\?|!)'
pattern2 = r"(\w+|#\d|\?|!)"
pattern3 = r'(#\\d\\w+\\?!)'
pattern4 = r'\\s+'

pprint(regexp_tokenize(my_string, pattern2))


# Regex with NLTK tokenization
tweets = ['This is the best #nlp exercise ive found online! #python',
          '#NLP is super fun! <3 #learning',
          'Thanks @datacamp :) #nlp #python']

from nltk.tokenize import regexp_tokenize, TweetTokenizer

# Define a regex pattern to find hashtags: pattern1
pattern1 = r"#\w+"

# Use the pattern on the first tweet in the tweets list
hashtags = regexp_tokenize(tweets[0], pattern1)
print(hashtags)

# write a pattern that matches both mentions (@) and hashtags
pattern2 = r"[@|#]\w+"

# Use the pattern on the last tweet in the tweets list
mentions_hashtags = regexp_tokenize(tweets[-1], pattern2)
print(mentions_hashtags)


# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)


# Non-ascii tokenization
german_text = 'Wann gehen wir Pizza essen? 🍕 Und fährst du mit Über? 🚕'

# Tokenize and print all words in german_text
all_words = word_tokenize(german_text)
print(all_words)

# Tokenize and print only capital words
capital_words = r"[A-ZÜ]\w+"
print(regexp_tokenize(german_text, capital_words))

# Tokenize and print only emoji
emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
print(regexp_tokenize(german_text, emoji))

# Charting word length with NLTK
import matplotlib.pyplot as plt

# Split the script into lines: lines
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(s, '\w+') for s in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.figure(figsize=(8, 8))
plt.hist(line_num_words)
plt.title('# of words per line in holy_grail')

