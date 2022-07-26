import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

marketing = pd.read_csv('./datasets/marketing.csv')

"""Let's explore our data.

**First let's print the first five rows**
"""

print(marketing.head())

"""**Now, we print a summary of the daatset**"""

print(marketing.describe())

"""**For last, we print the non.missing values and data type of all columns**"""

print(marketing.info())

"""**We are going to convert the data to the correct data types**"""

print(marketing['is_retained'].dtype)

marketing['is_retained'] = marketing['is_retained'].astype('bool')
print(marketing['is_retained'].dtype)

"""**Let's add a new column called *channel_code* which maps the values in the *subscribing_channel* column to a numeric scale using *channel_dict***"""

channel_dict = {"House Ads": 1,
                "Instagram": 2,
               "Facebook": 3,
               "Email": 4,
               "Push": 5}

marketing['channel_code'] = marketing['subscribing_channel'].map(channel_dict)

"""**Add a new column, *is_correct_lang*, which is *'Yes'* if the user was shown the ad in their preferred language, *'No'* otherwise.**"""

marketing['is_correct_lang'] = np.where(marketing['language_preferred'] == marketing['language_displayed'], 'Yes', 'No')

print(marketing.head())

"""**Currently the date columns are treated as Objects, let's change them to date type**"""

marketing['date_served'] = pd.to_datetime(marketing['date_served'])
marketing['date_subscribed'] = pd.to_datetime(marketing['date_subscribed'])
marketing['date_canceled'] = pd.to_datetime(marketing['date_canceled'])

print(marketing.info())

"""**Let's get the day of the week for every date type column**"""

marketing['date_served_dow'] = marketing['date_served'].dt.dayofweek
marketing['date_subscribed_dow'] = marketing['date_subscribed'].dt.dayofweek
marketing['date_canceled_dow'] = marketing['date_canceled'].dt.dayofweek

print(marketing.head())

"""## Some Exploratory Analysis of the Dataset

**Let's explore how many users are seeing the marketing assests each day**
"""

daily_users = marketing.groupby(['date_served'])['user_id'].nunique()
print(daily_users.head())

"""**It would look better in a plot. Let's plot it**"""

daily_users.plot()

plt.title('Daily Users')
plt.ylabel('Number of Users')
plt.xlabel('Date')

plt.xticks(rotation = 45)

plt.show()