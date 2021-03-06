import pandas as pd
import numpy as np

'''
# Feature selection
 - Selecting features to be used for modeling
 - Doesn't create new features
 - Improve model's performance
 
Identifying areas for feature selection
 - Take an exploratory look at the post-feature engineering hiking dataset.
'''
hiking = pd.read_json('./datacamp_repo/ML_Scientist_Career_Track/08_Preprocessing for Machine Learning in Python/'
                      'data/hiking.json')
hiking.head()


'''
# Removing redundant features
 - Remove noisy features
 - Remove correlated features
    - Statistically correlated: features move together directionally
    - Linear models assume feature independence
    - Pearson correlation coefficient
 - Remove duplicated features
'''

'''
# Selecting relevant features
Now let's identify the redundant columns in the volunteer dataset and perform feature selection on the dataset to 
return a DataFrame of the relevant features.

For example, if you explore the volunteer dataset in the console, you'll see three features which are related to location:
locality, region, and postalcode. They contain repeated information, so it would make sense to keep only one of the features.

There are also features that have gone through the feature engineering process: columns like Education and 
Emergency Preparedness are a product of encoding the categorical variable category_desc, 
so category_desc itself is redundant now.

Take a moment to examine the features of volunteer in the console, and try to identify the redundant features.
'''
volunteer = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/08_Preprocessing for Machine Learning in Python/'
                        'data/volunteer_sample.csv')
volunteer.dropna(subset=['category_desc'], axis=0, inplace=True)
volunteer.head()

print(volunteer.columns)

# Create a list of redundant column names to drop
to_drop = ["locality", "region", "category_desc", "created_date", "vol_requests"]

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of the new dataset
print(volunteer_subset.head())


'''
Checking for correlated features
Let's take a look at the wine dataset again, which is made up of continuous, numerical features. 
Run Pearson's correlation coefficient on the dataset to determine which columns are good candidates for eliminating. 
Then, remove those columns from the DataFrame.
'''
wine = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/08_Preprocessing for Machine Learning in Python/'
                   'data/wine_sample.csv')
wine.head()

# Print out the column correlations of the wine dataset
print(wine.corr())

# Take a minute to find the column where the correlation value is greater than 0.75 at least twice
to_drop = "Flavanoids"

# Drop that column from the DataFrame
wine = wine.drop(to_drop, axis=1)


# Selecting features using text vectors
'''
# Exploring text vectors, part 1
Let's expand on the text vector exploration method we just learned about, using the volunteer dataset's 
title tf/idf vectors. In this first part of text vector exploration, 
we're going to add to that function we learned about in the slides. We'll return a list of numbers with the function. 
In the next exercise, we'll write another function to collect the top words across all documents, extract them, 
and then use that list to filter down our text_tfidf vector.
'''
vocab_csv = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/08_Preprocessing for Machine Learning in Python/'
                        'data/vocab.csv', index_col=0).to_dict()
vocab = vocab_csv['0']
volunteer = volunteer[['category_desc', 'title']]
volunteer = volunteer.dropna(subset=['category_desc'], axis=0)

from sklearn.feature_extraction.text import TfidfVectorizer

# Take the title text
title_text = volunteer['title']

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)


# Add in the rest of the parameters
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))

    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]: zipped[i] for i in vector[vector_index].indices})

    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]


# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_, text_tfidf, vector_index=8, top_n=3))


'''
# Exploring text vectors, part 2
Using the function we wrote in the previous exercise, we're going to extract the top words from each document in 
the text vector, return a list of the word indices, and use that list to filter the text vector down to those top words.
'''


def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
        # here we'll call the function from the previous exercise,
        # and extend the list we're creating
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, top_n=3)

# By converting filtered_words back to a list,
# we can use it to filter the columns in the text vector
filtered_text = text_tfidf[:, list(filtered_words)]


'''
# Training Naive Bayes with feature selection
Let's re-run the Naive Bayes text classification model we ran at the end of chapter 3, 
with our selection choices from the previous exercise, on the volunteer dataset's title and category_desc columns.
'''
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
y = volunteer['category_desc']

# Split the dataset according to the class distribution of category_desc,
# using the filtered_text vector
X_train, X_test, y_train, y_test = train_test_split(filtered_text.toarray(), y, stratify=y)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))


'''
# Dimensionality reduction
 - Unsupervised learning method
 - Combine/decomposes a feature space
 - Feature extraction
 - Principal component analysis
    - Linear transformation to uncorrelated space
    - Captures as much variance as possible in each component

 - PCA caveats
    - Difficult to interpret components
    - End of preprocessing journey
'''

'''

Using PCA
Let's apply PCA to the wine dataset, to see if we can get an increase in our model's accuracy.
'''
wine = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/08_Preprocessing for Machine Learning in Python/'
                   'data/wine_types.csv')
wine.head()

from sklearn.decomposition import PCA

# Set up PCA and the X vector for dimensionality reduction
pca = PCA()
wine_X = wine.drop('Type', axis=1)

# Apply PCA to the wine dataset X vector
transformed_X = pca.fit_transform(wine_X)

# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)


'''
# Training a model with PCA
Now that we have run PCA on the wine dataset, let's try training a model with it.
'''
from sklearn.neighbors import KNeighborsClassifier

y = wine['Type']

knn = KNeighborsClassifier()

# Split the transformed X and the y labels into training and test sets
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(transformed_X, y)

# Fit knn to the training data
knn.fit(X_wine_train, y_wine_train)

# Score knn on the test data and print it out
print(knn.score(X_wine_test, y_wine_test))

