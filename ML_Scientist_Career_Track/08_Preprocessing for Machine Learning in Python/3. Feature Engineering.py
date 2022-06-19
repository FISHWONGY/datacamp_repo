import pandas as pd
import numpy as np

'''
# Feature engineering
 - Creation of new features based on existing features
 - Insight into relationships between features
 - Extract and expand data
 - Dataset-dependent

# Identifying areas for feature engineering
Take an exploratory look at the volunteer dataset, using the variable of that name. 
Which of the following columns would you want to perform a feature engineering task on?
'''
volunteer = pd.read_csv(
    '/Online course/datacamp_repo/ML_Scientist_Career_Track/08_Preprocessing for Machine Learning in Python/data/volunteer_opportunities.csv')
volunteer.head()


'''
# Encoding categorical variables
Encoding categorical variables - binary
Take a look at the hiking dataset. There are several columns here that need encoding, 
one of which is the Accessible column, which needs to be encoded in order to be modeled. 
Accessible is a binary feature, so it has two values - either Y or N - so it needs to be encoded into 1s and 0s. 
Use scikit-learn's LabelEncoder method to do that transformation.
'''
hiking = pd.read_json(
    '/Online course/datacamp_repo/ML_Scientist_Career_Track/08_Preprocessing for Machine Learning in Python/data/hiking.json')
hiking.head()

from sklearn.preprocessing import LabelEncoder

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking['Accessible'])

# Compare the two columns
hiking[['Accessible', 'Accessible_enc']].head()


'''
# Encoding categorical variables - one-hot
One of the columns in the volunteer dataset, category_desc, gives category descriptions for the volunteer opportunities listed. 
Because it is a categorical variable with more than two categories, we need to use one-hot encoding to transform 
this column numerically. Use Pandas' get_dummies() function to do so.
'''
# Transform the category_desc column
category_enc = pd.get_dummies(volunteer['category_desc'])

# Take a look at the encoded columns
print(category_enc.head())


# Engineering numerical features
'''
# Engineering numerical features - taking an average
A good use case for taking an aggregate statistic to create a new feature is to take the mean of columns. 
Here, you have a DataFrame of running times named running_times_5k. For each name in the dataset, take the mean 
of their 5 run times.
'''
running_times_5k = pd.read_csv(
    '/Online course/datacamp_repo/ML_Scientist_Career_Track/08_Preprocessing for Machine Learning in Python/data/running_times_5k.csv')

# Create a list of the columns to average
run_columns = ['run1', 'run2', 'run3', 'run4', 'run5']

# Use apply to create a mean column
running_times_5k['mean'] = running_times_5k.apply(lambda row: row[run_columns].mean(), axis=1)

# Take a look at the results
print(running_times_5k)


'''
# Engineering numerical features - datetime
There are several columns in the volunteer dataset comprised of datetimes. 
Let's take a look at the start_date_date column and extract just the month to use as a feature for modeling.
'''
# First, convert string column to date column
volunteer['start_date_converted'] = pd.to_datetime(volunteer['start_date_date'])

# Extract just the month from the converted column
volunteer['start_date_month'] = volunteer['start_date_converted'].apply(lambda row: row.month)

# Take a look at the converted and new month columns
volunteer[['start_date_converted', 'start_date_month']].head()


'''
# Text classification
Engineering features from strings - extraction
The Length column in the hiking dataset is a column of strings, but contained in the column is the mileage for the hike. 
We're going to extract this mileage using regular expressions, and then use a lambda in Pandas to apply the extraction 
to the DataFrame.
'''
import re


# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r'\d+\.\d+')

    if length == None:
        return

    # Search the text for matches
    mile = re.match(pattern, length)

    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))


# Apply the function to the Length column and take a look at both columns
hiking['Length_num'] = hiking['Length'].apply(lambda row: return_mileage(row))
hiking[['Length', 'Length_num']].head()


'''
# Engineering features from strings - tf/idf
Let's transform the volunteer dataset's title column into a text vector, to use in a prediction task in the next exercise.
'''
from sklearn.feature_extraction.text import TfidfVectorizer

# Need to drop NaN for train_test_split
volunteer = pd.read_csv('./dataset/volunteer_opportunities.csv')
volunteer = volunteer.dropna(subset=['category_desc'], axis=0)

# Take the title text
title_text = volunteer['title']

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)


'''
# Text classification using tf/idf vectors
Now that we've encoded the volunteer dataset's title column into tf/idf vectors, 
let's use those vectors to try to predict the category_desc column.
'''
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

# Split the dataset according to the class distribution of category_desc
y = volunteer['category_desc']
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify=y)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))