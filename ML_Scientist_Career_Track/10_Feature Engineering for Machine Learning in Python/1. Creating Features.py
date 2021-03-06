import pandas as pd
import numpy as np

# Why generate features?
'''
Different types of data
 - Continuous: either integers (or whole numbers) or floats (decimals)
 - Categorical: one of a limited set of values, e.g., gender, country of birth
 - Ordinal: ranked values often with no details of distance between them
 - Boolean: True/False values
 - Datetime: dates and times
'''

'''
# Getting to know your data
You will be working with a modified subset of the Stackoverflow survey response data in the first three chapters of 
this course. This data set records the details, and preferences of thousands of users of the StackOverflow website.
'''
# Import the data
so_survey_df = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                           '10_Feature Engineering for Machine Learning in Python/data/Combined_DS_v10.csv')

# Print the first five rows of the DataFrame
print(so_survey_df.head())

# Print the data type of each column
print(so_survey_df.dtypes)


'''
# Selecting specific data types

Often a data set will contain columns with several different data types (like the one you are working with). 
The majority of machine learning models require you to have a consistent data type across features. 
Similarly, most feature engineering techniques are applicable to only one type of data at a time. 
For these reasons among others, you will often want to be able to access just the columns of certain types when working 
with a DataFrame.
'''
# Create subset of only the numberic columns
so_numeric_df = so_survey_df.select_dtypes(include=['int', 'float'])

# Print the column names contained in so_numeric_df
print(so_numeric_df.columns)


'''
# Dealing with categorical features
 - Encoding categorical features
    - One-hot encoding
    - Dummy encoding

- One-hot vs. dummies
    - One-hot encoding: Explainable features
    - Dummy encoding: Necessary information without duplication
'''

'''
# One-hot encoding and dummy variables
To use categorical variables in a machine learning model, you first need to represent them in a quantitative way. 
The two most common approaches are to one-hot encode the variables using or to use dummy variables. 
In this exercise, you will create both types of encoding, and compare the created column sets.
'''
# Convert the Country column to a one hot encoded DataFrame
one_hot_encoded = pd.get_dummies(so_survey_df, columns=['Country'], prefix='OH')

# Print the columns names
print(one_hot_encoded.columns)

# Convert the Country column to a one hot encoded DataFrame
dummy = pd.get_dummies(so_survey_df, columns=['Country'], drop_first=True, prefix='DM')

# Print the columns names
print(dummy.columns)


'''
# Dealing with uncommon categories
Some features can have many different categories but a very uneven distribution of their occurrences. 
Take for example Data Science's favorite languages to code in, some common choices are Python, R, and Julia, 
but there can be individuals with bespoke choices, like FORTRAN, C etc. 
In these cases, you may not want to create a feature for each value, but only the more common occurrences.
'''
# Create a series out of the Country columns
countries = so_survey_df.Country

# Get the counts of each category
country_counts = countries.value_counts()

# Print the count values for each category
print(country_counts)


# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Print the top 5 rows in the mask series
print(mask.head())

# Label all other categories as Other
countries[mask] = 'Other'

# Print the updated category counts
print(countries.value_counts())


###
# Numeric variables
'''
# Binarizing columns
While numeric values can often be used without any feature engineering, there will be cases when some form of 
manipulation can be useful. For example on some occasions, 
you might not care about the magnitude of a value but only care about its direction, 
or if it exists at all. In these situations, you will want to binarize a column. 
In the so_survey_df data, you have a large number of survey respondents that are working voluntarily (without pay). 
You will create a new column titled Paid_Job indicating whether each person is paid (their salary is greater than zero).
'''
# Create the Paid_Job column filled with zeros
so_survey_df['Paid_Job'] = 0

# Replace all the Paid_Job values where ConvertedSalary is > 0
so_survey_df.loc[so_survey_df['ConvertedSalary'] > 0, 'Paid_Job'] = 1

# Print the first five rows of the columns
so_survey_df[['Paid_Job', 'ConvertedSalary']].head()


'''
# Binning values
For many continuous values you will care less about the exact value of a numeric column, 
but instead care about the bucket it falls into. This can be useful when plotting values, 
or simplifying your machine learning models. 
It is mostly used on continuous variables where accuracy is not the biggest concern e.g. age, height, wages.

Bins are created using pd.cut(df['column_name'], bins) where bins can be an integer specifying the number of 
evenly spaced bins, or a list of bin boundaries.
'''
# Bin the continouos variable ConvertedSalary into 5 bins
so_survey_df['equal_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins=5)

# Print the first 5 rows of the equal_binned column
so_survey_df[['equal_binned', 'ConvertedSalary']].head()

# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]

# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Bin the continous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins=bins, labels=labels)

# Print the first 5 rows of the boundary_binned column
so_survey_df[['boundary_binned', 'ConvertedSalary']].head()

