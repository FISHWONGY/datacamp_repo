import pandas as pd

'''
What is data preprocessing?
Data Preprocessing
 - Beyond cleaning and exploratory data analysis
 - Prepping data for modeling
 - Modeling in python requires numerical input
'''

'''
Missing data - columns
We have a dataset comprised of volunteer information from New York City. 
The dataset has a number of features, but we want to get rid of features that have at least 3 missing values.
'''
volunteer = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/08_Preprocessing for Machine Learning in Python/'
                        'data/volunteer_opportunities.csv')
volunteer.head()
volunteer.info()
volunteer.dropna(axis=1, thresh=3).shape
volunteer.shape


'''
# Missing data - rows

Taking a look at the volunteer dataset again, we want to drop rows where the category_desc column values are missing. 
We're going to do this using boolean indexing, by checking to see if we have any null values, and then 
filtering the dataset so that we only have rows with those values.
'''
# Check how many values are missing in the category_desc column
print(volunteer['category_desc'].isnull().sum())

# Subset the volunteer dataset
volunteer_subset = volunteer[volunteer['category_desc'].notnull()]

# Print out the shape of the subset
print(volunteer_subset.shape)


'''
Working with data types
dtypes in pandas
object: string/mixed types
int64: integer
float64: float
datetime64 (or timedelta): datetime
'''
volunteer.dtypes


'''
Converting a column type
If you take a look at the volunteer dataset types, you'll see that the column hits is type object.
But, if you actually look at the column, you'll see that it consists of integers. Let's convert that column to type int.
'''
# Print the head of the hits column
print(volunteer['hits'].head())

# Convert the hits column to type int
volunteer['hits'] = volunteer['hits'].astype(int)

# Look at the dtypes of the dataset
print(volunteer.dtypes)


'''
Class distribution

Stratified sampling
 - A way of sampling that takes into account the distribution of classes or features in your dataset
'''

'''
Class imbalance
In the volunteer dataset, we're thinking about trying to predict the category_desc variable 
using the other features in the dataset. 
First, though, we need to know what the class distribution (and imbalance) is for that label.
'''
volunteer['category_desc'].value_counts()

'''
Stratified sampling
We know that the distribution of variables in the category_desc column in the volunteer dataset is uneven. 
If we wanted to train a model to try to predict category_desc, we would want to train the model on a sample of 
data that is representative of the entire dataset. 
Stratified sampling is a way to achieve this.
'''

from sklearn.model_selection import train_test_split

# Create a data with all columns except category_desc
volunteer_X = volunteer.dropna(subset=['category_desc'], axis=0)

# Create a category_desc labels dataset
volunteer_y = volunteer.dropna(subset=['category_desc'], axis=0)[['category_desc']]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(volunteer_X, volunteer_y, stratify=volunteer_y)

# Print out the category_desc counts on the training y labels
print(y_train['category_desc'].value_counts())

