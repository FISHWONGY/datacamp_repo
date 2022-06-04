import pandas as pd

movies = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/'
                     'data/IMDB_sample.csv')

# Find the number of positive and negative reviews
print('Number of positive and negative reviews: ', movies.label.value_counts())

# Find the proportion of positive and negative reviews
print('Proportion of positive and negative reviews: ', movies.label.value_counts() / len(movies))

length_reviews = movies.review.str.len()

# How long is the longest review
print(max(length_reviews))

# How long is the shortest review
print(min(length_reviews))


# Detecting the sentiment of Tale of Two Cities
# Import the required packages
from textblob import TextBlob

with open(
        '/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/data'
        '/two_cities.txt.txt', 'r') as f:
    two_cities = f.read()

# Create a textblob object
blob_two_cities = TextBlob(two_cities)

# Print out the sentiment
print(blob_two_cities.sentiment)

# Create a textblob object
annak = 'Happy families are all alike; every unhappy family is unhappy in its own way'
catcher = "If you really want to hear about it,the first thing you'll probably want to know is where I was born, " \
          "and what my lousy childhood was like, and how my parents were occupied and all before they had me, " \
          "and all that David Copperfield kind of crap, but I don't feel like going into it, " \
          "if you want to know the truth."
blob_annak = TextBlob(annak)
blob_catcher = TextBlob(catcher)

# Print out the sentiment
print('Sentiment of annak: ', blob_annak.sentiment)
print('Sentiment of catcher: ', blob_catcher.sentiment)

# Titanic
with open(
        '/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/data'
        '/titanic.txt', 'r') as f:
    titanic = f.read()

# Create a textblob object
blob_titanic = TextBlob(titanic)

# Print out its sentiment
print(blob_titanic.sentiment)


# Word Cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open(
        '/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/data'
        '/east_of_eden.txt', 'r') as f:
    east_of_eden = f.read()

# Generate the word cloud from the east_of_eden string
cloud_east_of_eden = WordCloud(background_color="white").generate(east_of_eden)

# Create a figure of the generated cloud
plt.imshow(cloud_east_of_eden, interpolation='bilinear')
plt.axis('off')
# Display the figure
plt.show()

# Create and generate a word cloud image
my_stopwords = {'up', 'few', 'very', 'http', 'his', 'hence', 'com', 'how', 'but', 'yourselves', "we've", 'its', 'so', 'if', 'get', "he'll", 'also', 'otherwise', 'their', "they'll", 'what', 'be', 'out', "shan't", 'above', 'are', 'it', "she'd", 'some', 'whom', 'therefore', 'who', 'why', "he's", 'against', 'doing', 'not', "can't", "you're", 'a', 'hers', 'no', 'him', 'to', 'such', 'would', 'however', 'at', "how's", 'most', "aren't", 'as', "mustn't", 'did', 'here', "she's", "where's", 'ought', 'myself', "won't", "wouldn't", 'could', 'after', "when's", 'any', "wasn't", 'films', 'because', "why's", 'off', 'we', "doesn't", 'you', 'in', 'them', 'both', 'our', 'itself', 'theirs', 'from', 'should', 'before', 'between', 'too', 'with', 'am', 'her', "we're", 'since', "they'd", 'an', "couldn't", 'ourselves', "weren't", 'these', 'which', 'k', 'further', 'once', 'have', 'has', 'for', "i'll", 'br', 'watch', "i'm", 'all', 'where', 'does', 'ever', 'until', 'through', 'during', "you'll", 'been', "we'll", 'he', 'yourself', 'by', "who's", 'over', 'is', 'movies', "that's", 'below', "he'd", 'were', 'being', "shouldn't", 'than', 'ours', "here's", "i've", 'more', 'about', "it's", 'on', 'same', 'she', "she'll", 'herself', 'yours', "i'd", 'just', "they've", 'i', 'your', 'themselves', 'like', 'under', "they're", 'that', 'again', "we'd", "hasn't", 'me', 'nor', 'shall', 'www', 'film', 'had', 'or', 'they', "isn't", "haven't", "didn't", 'can', 'while', "hadn't", "there's", 'other', 'those', 'cannot', "don't", 'into', "you'd", 'of', 'was', "what's", 'else', 'own', 'my', "let's", 'this', 'there', 'down', 'do', 'having', 'each', 'r', 'then', 'only', 'when', 'and', "you've", 'movie', 'himself',
                'the'}

with open(
        '/Volumes/My Passport for Mac/Python/Online course/datacamp/Sentiment Analysis in Python/data'
        '/descriptions.txt', 'r') as f:
    descriptions = f.read()

my_cloud = WordCloud(background_color='white', stopwords=my_stopwords).generate(descriptions)

# Display the generated wordcloud image
plt.imshow(my_cloud, interpolation='bilinear')
plt.axis("off")

# Don't forget to show the final image
plt.show()