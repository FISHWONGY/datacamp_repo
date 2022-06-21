import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Non-Negative matrix factorization (NMF)
'''
NMF = Non-negative matrix factorization
    Dimension reduction technique
    NMF models are interpretable (unlike PCA)
    Easy to interpret means easy to explain
    However, all sample features must be non-negative 

NMF components
    Just like PCA has principal components
    Dimension of components = dimension of samples
    Entries are non-negative
    Can be used to reconstruct the samples
    Combine feature values with components

Sample reconstruction
    Multiply components by feature values, and add up
    Can also be expressed as a product of matrices
    This is the "Matrix Factorization" in "NMF"
'''

# NMF applied to Wikipedia articles
'''
In the video, you saw NMF applied to transform a toy word-frequency array. Now it's your turn to apply NMF, 
this time using the tf-idf word-frequency array of Wikipedia articles, given as a csr matrix articles. 
Here, fit the model and transform the articles. In the next exercise, you'll explore the result.
'''
from scipy.sparse import csc_matrix

documents = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                        '02_Unsupervised Learning in Python/data/wikipedia-vectors.csv', index_col=0)
titles = documents.columns
articles = csc_matrix(documents.values).T

from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features)


# NMF features of the Wikipedia articles
'''
Now you will explore the NMF features you created in the previous exercise. 
A solution to the previous exercise has been pre-loaded, so the array nmf_features is available. 
Also available is a list titles giving the title of each Wikipedia article.

When investigating the features, notice that for both actors, the NMF feature 3 has by far the highest value. 
This means that both articles are reconstructed using mainly the 3rd NMF component. 
In the next video, you'll see why: NMF components represent topics (for instance, acting!).
'''


# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])


# NMF reconstructs samples
'''
In this exercise, you'll check your understanding of how NMF reconstructs samples from its components using the NMF feature values. 
On the right are the components of an NMF model. If the NMF feature values of a sample are [2, 1], then which of the 
following is most likely to represent the original sample? A pen and paper will help here! 
You have to apply the same technique Ben used in the video to reconstruct the sample [0.1203 0.1764 0.3195 0.141].
'''
sample_feature = np.array([2, 1])
components = np.array([[1. , 0.5, 0. ],
                       [0.2, 0.1, 2.1]])

np.matmul(sample_feature.T, components)


# NMF learns interpretable parts
'''
In the video, you learned when NMF is applied to documents, the components correspond to topics of documents, and the 
NMF features reconstruct the documents from the topics. 
Verify this for yourself for the NMF model that you built earlier using the Wikipedia articles. 
Previously, you saw that the 3rd NMF feature value was high for the articles about actors Anne Hathaway and Denzel Washington. 
In this exercise, identify the topic of the corresponding NMF component.

The NMF model you built earlier is available as model, while words is a list of the words that label the columns of the 
word-frequency array.

After you are done, take a moment to recognise the topic that the articles about Anne Hathaway and Denzel Washington have in common!
'''



words = []
with open('./datacamp_repo/ML_Scientist_Career_Track/'
          '02_Unsupervised Learning in Python/data/wikipedia-vocabulary-utf8.txt') as f:
    words = f.read().splitlines()

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())


# Explore the LED digits dataset
'''
In the following exercises, you'll use NMF to decompose grayscale images into their commonly occurring patterns. 
Firstly, explore the image dataset and see how it is encoded as an array. 
You are given 100 images as a 2D array samples, where each row represents a single 13x8 image. 
The images in your dataset are pictures of a LED digital display.
'''
df = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                 '02_Unsupervised Learning in Python/data/lcd-digits.csv', header=None)
df.head()

samples = df.values

# Select the 0th row: digit
digit = samples[0]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13, 8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()


# NMF learns the parts of images
'''
Now use what you've learned about NMF to decompose the digits dataset. You are again given the digit images as a 2D array samples. 
This time, you are also provided with a function show_as_image() that displays the image encoded by any 1D array:
'''
def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()


# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0]

# Print digit_features
print(digit_features)


# PCA doesn't learn parts
'''
Unlike NMF, PCA doesn't learn the parts of things. 
Its components do not correspond to topics (in the case of documents) or to parts of images, when trained on images. 
Verify this for yourself by inspecting the components of a PCA model fit to the dataset of LED digit images from 
the previous exercise. The images are available as a 2D array samples. 
Also available is a modified version of the show_as_image() function which colors a pixel red if the value is negative.

After submitting the answer, notice that the components of PCA do not represent meaningful parts of images of LED digits!
'''


from sklearn.decomposition import PCA

# Createa PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)


# Building recommender systems using NMF
'''
Finding similar articles
    Engineer at a large online newspaper
    Task: recommand articles similar to article being read by customer
    Similar articles should have similar topics

Strategy
    Apply NMF to the word-frequency array
    NMF feature values describe the topics, so similar documents have similar NMF feature values
    Compare NMF feature values?

Versions of articles
    Different versions of the same document have same topic proportions
    exact feature values may be different! E.g., because one version uses many meaningless words
    But all versions lie on the same line through the origin
'''

# Which articles are similar to 'Cristiano Ronaldo'?
'''
In the video, you learned how to use NMF features and the cosine similarity to find similar articles. Apply this to your 
NMF model for popular Wikipedia articles, by finding the articles most similar to the article about the footballer Cristiano Ronaldo. 
The NMF features you obtained earlier are available as nmf_features, while titles is a list of the article titles.
'''


from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display thouse with the largest cosine similarity
print(similarities.nlargest())


# Recommend musical artists part I
'''
In this exercise and the next, you'll use what you've learned about NMF to recommend popular music artists! 
You are given a sparse array artists whose rows correspond to artists and whose columns correspond to users. 
The entries give the number of times each artist was listened to by each user.

In this exercise, build a pipeline and transform the array into normalized NMF features. 
The first step in the pipeline, MaxAbsScaler, transforms the data so that all users have the same influence on the model, 
regardless of how many different artists they've listened to. 
In the next exercise, you'll use the resulting normalized NMF features for recommendation!
'''


from scipy.sparse import coo_matrix

df = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                 '02_Unsupervised Learning in Python/data/scrobbler-small-sample.csv')
artists1 = df.sort_values(['artist_offset', 'user_offset'], ascending=[True, True])
row_ind = np.array(artists1['artist_offset'])
col_ind = np.array(artists1['user_offset'])
data1 = np.array(artists1['playcount'])
artists = coo_matrix((data1, (row_ind, col_ind)))

print(artists)

from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)


# Recommend musical artists part II
'''
Suppose you were a big fan of Bruce Springsteen - which other musicial artists might you like? 
Use your NMF features from the previous exercise and the cosine similarity to find similar musical artists. 
A solution to the previous exercise has been run, so norm_features is an array containing the normalized NMF features as rows. 
The names of the musical artists are available as the list artist_names.
'''
df = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                 '02_Unsupervised Learning in Python/data/artists.csv', header=None)
artist_names = df.values.reshape(111).tolist()

print(artist_names)

# Createa DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())


