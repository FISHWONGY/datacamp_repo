import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

marketing = pd.read_csv('./datasets/marketing.csv')

print(marketing.head())

"""#### Conversion Rate

% of people we marketed to who ultimately converted to our product

\begin{equation*}
\text{Conversion Rate} = \frac{\text{Number of people who convert}}{\text{Total Number of people we marketed to}}
\end{equation*}
"""

subscribers = marketing[marketing['converted'] == True]['user_id'].nunique()
total = marketing['user_id'].nunique()
conv_rate = subscribers / total

print(round(conv_rate *100,2),'%')

"""#### Retention Rate

% of people that remain subscribed after a certain period of time

\begin{equation*}
\text{Retention Rate} = \frac{\text{Number of people who remain subscribed}}{\text{Total Number of people who converted}}
\end{equation*}
"""

retained = marketing[marketing['is_retained'] == True]['user_id'].nunique()

subscribers = marketing[marketing['converted'] == True]['user_id'].nunique()

retention = retained / subscribers

print(round(retention*100, 2), '%')

"""### Customer Segmentation

**Common ways to segment audiences**
- Age
- Gender
- Location
- Past interactions with the business
- Marketing channels the user interacted with

**Subset of only House Ads**
"""

house_ads = marketing[marketing['subscribing_channel'] == 'House Ads']

print(house_ads.head())

"""**Percentage of user retained using *'House Ads'***"""

retained = house_ads[house_ads['is_retained'] == True]['user_id'].nunique()

subscribers = house_ads[house_ads['converted'] == True]['user_id'].nunique()

retention = retained / subscribers

print(round(retention*100, 2), '%')

"""**Retention by channel**"""

retained = marketing[marketing['is_retained'] == True].groupby(['subscribing_channel'])['user_id'].nunique()
print(retained)

"""**Converted by channel**"""

subscribers = marketing[marketing['converted'] == True].groupby(['subscribing_channel'])['user_id'].nunique()
print(subscribers)

"""**Retention Rate by channel**"""

channel_retention_rate = (retained/subscribers)*100
print(channel_retention_rate)

"""**Customer Segmentation by language**"""

english_speakers = marketing[marketing['language_displayed'] == 'English']

total = english_speakers['user_id'].nunique()

subscribers = english_speakers[english_speakers['converted'] == True]['user_id'].nunique()

conversion_rate = subscribers/total
print('English speaker conversion rate:', round(conversion_rate*100,2), '%')

total = marketing.groupby(['language_displayed'])['user_id'].nunique()
subscribers = marketing[marketing['converted'] == True].groupby(['language_displayed'])['user_id'].nunique()

language_conversion_rate = subscribers/total
print(language_conversion_rate)

"""**Aggregating by date**"""

total = marketing.groupby(['date_subscribed'])['user_id'].nunique()
retained = marketing[marketing['is_retained'] == True].groupby(['date_subscribed'])['user_id'].nunique()

daily_retention_rate = retained/total

print(daily_retention_rate.head())

"""**Plotting the previous results**"""

language_conversion_rate.plot(kind = 'bar')

plt.title('Conversion rate by language\n', size = 16)
plt.xlabel('Language', size = 14)
plt.ylabel('Conversion rate (%)', size = 14)

plt.show()

"""Now, let's first convert the series into a DF"""

daily_retention_rate = pd.DataFrame(daily_retention_rate.reset_index())

daily_retention_rate.columns=['date_subscribed',
                              'retention_rate']

print(daily_retention_rate.head())

daily_retention_rate.plot('date_subscribed', 'retention_rate')
plt.title('Daily Subscriber Quality\n', size=16)
plt.ylabel('1-month retention rate (%)', size=14)
plt.xlabel('Date', size = 14)

plt.ylim(0)

plt.show()

"""**Marketing channels accross Age groups**"""

channel_age = marketing.groupby(['marketing_channel', 'age_group'])['user_id'].count()

channel_age_df = pd.DataFrame(channel_age.unstack(level=1))
print(channel_age_df.head())

channel_age_df.plot(kind='bar')
plt.title('Channel preferences by Age group')
plt.xlabel('Age group')
plt.ylabel('Channel')
plt.legend(loc='upper right',
           labels = channel_age_df.columns.values)

plt.show()

"""**Grouping and counting by multiple columns**"""

retention_total = marketing.groupby(['date_subscribed',
                                     'subscribing_channel'])['user_id'].nunique()

print(retention_total.head())

retention_subs = marketing[marketing['is_retained'] == True].groupby(['date_subscribed', 
                                                                      'subscribing_channel'])['user_id'].nunique()
print(retention_subs.head())

retention_rate = retention_subs/retention_total
retention_rate_df = pd.DataFrame(retention_rate.unstack(level = 1))

retention_rate_df.plot()

plt.title('Retention Rate by Subscribing Channel')
plt.xlabel('Date Subscribed')
plt.ylabel('Retention Rate (%)')

plt.legend(loc='upper right',
           labels = retention_rate_df.columns.values)

plt.show()