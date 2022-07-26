import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

marketing = pd.read_csv('./datasets/marketing.csv')

"""**To avoid repetition, we create a function that calculates conversion rate**"""


def conversion_rate(dataframe, column_names):
    #Total number of converted users
    column_conv = dataframe[dataframe['converted'] == True].groupby(column_names)['user_id'].nunique()
    
    #Total number of users
    column_total = dataframe.groupby(column_names)['user_id'].nunique()
    
    #Conversion Rate
    conversion_rate = column_conv/column_total
    
    #Fill missing values with 0
    conversion_rate = conversion_rate.fillna(0)
    
    return conversion_rate

"""**Let's test the function calculating some conversion rate**"""

age_group_conv = conversion_rate(marketing, ['date_served', 'age_group'])

print(age_group_conv.head())

age_group_df = pd.DataFrame(age_group_conv.unstack(level=1))

print(age_group_df.head())

age_group_df.plot()
plt.title('Conversion rate by age group\n', size = 16)
plt.ylabel('Conversion rate', size = 14)
plt.xlabel('Age group', size = 14)
plt.show()

"""**Now a function for plotting**"""

def plotting_conv(dataframe):
    for column in dataframe:
        plt.plot(dataframe.index, dataframe[column])
        plt.title('Daily ' + str(column) + ' conversion rate\n', size = 16)
        plt.ylabel('Conversion rate', size = 14)
        plt.xlabel('Date', size = 14)
        plt.show()
        plt.clf()

"""**Time to use both functions**"""

age_group_conv = conversion_rate(marketing, ['date_served', 'age_group'])

age_group_df = pd.DataFrame(age_group_conv.unstack(level=1))

plotting_conv(age_group_df)

"""## Identifying inconsistencines"""

daily_conv_channel = conversion_rate(marketing, ['date_served', 'marketing_channel'])

print(daily_conv_channel.head())

daily_conv_channel = pd.DataFrame(daily_conv_channel.unstack(level = 1))

plotting_conv(daily_conv_channel)

marketing['date_served'] = pd.to_datetime(marketing['date_served'])
marketing['DoW_served'] = marketing['date_served'].dt.dayofweek

print(marketing.head())

DoW_conversion = conversion_rate(marketing, ['DoW_served', 'marketing_channel'])
DoW_df = pd.DataFrame(DoW_conversion.unstack(level = 1))

DoW_df.plot()
plt.title('Conversion rate by day of week\n')
plt.ylim(0)
plt.show()

house_ads = marketing[marketing['marketing_channel'] == 'House Ads']

conv_lang_channel = conversion_rate(house_ads, ['date_served', 'language_displayed'])

conv_lang_df = pd.DataFrame(conv_lang_channel.unstack(level = 1))

plotting_conv(conv_lang_df)

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# house_ads['is_correct_lang'] = np.where(house_ads['language_displayed'] == house_ads['language_preferred'], 'Yes', 'No')
# 
# language_check = house_ads.groupby(['date_served', 'is_correct_lang'])['user_id'].count()
# 
# language_check_df = pd.DataFrame(language_check.unstack(level=1)).fillna(0)

print(language_check_df)

language_check_df['pct'] = language_check_df['Yes'] / language_check_df.sum(axis = 1)
print(language_check_df.head())

plt.plot(language_check_df.index.values, language_check_df['pct'])
plt.show()

"""## Resolving Inconsistencies

**Now that we've determined that tre problem in the conversion rate was that the users received emails in another language, we are going to predict what was the lost given this issue**

**First, we are going to index the non-English conversion rates against English conversion rates in the time period before the language bug**
"""

#first, a new DF who contains data before the bug
house_ads_bug = house_ads[house_ads['date_served'] < '2018-01-11']

print(house_ads_bug.head())

lang_conv = conversion_rate(house_ads_bug, ['language_displayed'])

print(lang_conv)

# Index other language conversion rate against English
spanish_index = lang_conv['Spanish']/lang_conv['English']
arabic_index = lang_conv['Arabic']/lang_conv['English']
german_index = lang_conv['German']/lang_conv['English']

print("Spanish index:", spanish_index)
print("Arabic index:", arabic_index)
print("German index:", german_index)

"""**The previous data indicates how many times each language converts compared to English. For example, Spanish speaker clients, convert 1.68 times more than English speaker clients. Arabic 5 times and German 4 times**

**What we are trying to find is how many subscribers we would have expected if there had no been a language error**

**This DF is going to help us to calculate the expected number of subscribers**
"""

#Group the DataFrame by date_served and by language
#Count the number of unique clients
#Sum the number of converted clients
converted = house_ads.groupby(['date_served', 'language_preferred']).agg({'user_id':'nunique', 'converted': 'sum'})

print(converted.head())

converted_df = pd.DataFrame(converted.unstack(level = 1))

print(converted_df.head())

"""**The next step is to create a DF thet will estimate what daily conversion rates would have been if users were being served the correct language**"""

#Made just to fit the platform excercise
converted = converted_df
#Create a column with the English conversion rate between 11/01/2018 and 31/01/2018
converted['english_conv_rate'] = converted.loc['2018-01-11':'2018-01-31'][('converted', 'English')]
print(converted.tail())

#Create expected conversion rate for each language
converted['expected_spanish_rate'] = converted['english_conv_rate'] * spanish_index
converted['expected_arabic_rate'] = converted['english_conv_rate'] * arabic_index
converted['expected_german_rate'] = converted['english_conv_rate'] * german_index

print(converted.tail())

#Multiply the number of users bye the expected conversion rate
converted['expected_spanish_conv'] = converted['expected_spanish_rate'] * converted[('user_id', 'Spanish')] / 100
converted['expected_arabic_conv'] = converted['expected_arabic_rate'] * converted[('user_id', 'Arabic')] / 100
converted['expected_german_conv'] = converted['expected_german_rate'] * converted[('user_id', 'German')] / 100

print(converted.tail())

"""**It's time to calculate how many subscribers were lost due to the error**"""

#Create a DF containing data where date is between 11/01/2018 and 31/01/2018
converted = converted.loc['2018-01-11':'2018-01-31']

expected_subs = converted['expected_spanish_conv'].sum() + converted['expected_arabic_conv'].sum() + converted['expected_german_conv'].sum()

print("Expected Subscribers:", expected_subs)

actual_subs = converted[('converted', 'Spanish')].sum() + converted[('converted', 'Arabic')].sum() + converted[('converted', 'German')].sum()

print("Actual Subscribers:",actual_subs)

lost_subs = expected_subs - actual_subs

print("Lost Subscribers:",lost_subs)