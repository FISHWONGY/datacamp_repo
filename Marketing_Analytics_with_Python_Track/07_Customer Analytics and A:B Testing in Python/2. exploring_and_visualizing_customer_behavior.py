import pandas as pd
from datetime import timedelta

customer_data = pd.read_csv("./datasets/user_demographics_v1.csv")
app_purchases = pd.read_csv("./datasets/purchase_data_v1.csv")

customer_data.info()
app_purchases.info()

"""**Timedelta class**"""

current_date = pd.to_datetime('2018-03-17')
max_lapse_date = current_date - timedelta(days=14)

print(max_lapse_date)

"""**Parsing dates on import**"""

customer_demographics = pd.read_csv('./datasets/user_demographics_v1.csv', parse_dates = True, infer_datetime_format = True)

customer_demographics.info()

"""**Excercises**

Datetime reference link http://strftime.org/

"""

date_data_one = ['Saturday January 27, 2017', 'Saturday December 2, 2017']
# Provide the correct format for the date
date_data_one = pd.to_datetime(date_data_one, format= "%A %B %d, %Y")
print(date_data_one)

date_data_two = ['2017-01-01', '2016-05-03']
# Provide the correct format for the date
date_data_two = pd.to_datetime(date_data_two, format="%Y-%m-%d")
print(date_data_two)

date_data_three = ['08/17/1978', '01/07/1976']
# Provide the correct format for the date
date_data_three = pd.to_datetime(date_data_three, format="%m/%d/%Y")
print(date_data_three)

date_data_four = ['2016 March 01 01:56', '2016 January 4 02:16']
date_data_four = pd.to_datetime(date_data_four, format="%Y %B %d %H:%M")
print(date_data_four)

"""**Time Series graphs**"""

# Group the data and aggregate first_week_purchases
user_purchases = user_purchases.groupby(by=['reg_date', 'uid']).agg({'first_week_purchases': ['sum']})

# Reset the indexes
user_purchases.columns = user_purchases.columns.droplevel(level=1)
user_purchases.reset_index(inplace=True)

# Find the average number of purchases per day by first-week users
user_purchases = user_purchases.groupby(by=['reg_date']).agg({'first_week_purchases': ['mean']})
user_purchases.columns = user_purchases.columns.droplevel(level=1)
user_purchases.reset_index(inplace=True)

# Plot the results
user_purchases.plot(x='reg_date', y='first_week_purchases')
plt.show()

# Pivot the data
device_pivot = pd.pivot_table(user_purchases_device, values=['first_week_purchases'], columns=['device'], index=['reg_date'])
print(device_pivot.head())

# Plot the average first week purchases for each country by registration date
country_pivot.plot(x='reg_date', y=['USA', 'CAN', 'FRA', 'BRA', 'TUR', 'DEU'])
plt.show()

"""**Understanding and Visualizing Trends**

- **Trailing Average:** smoothing technique that averages over **lagging window**
    - Reveal hidden trends by smoothing out seasonality
    - Average across the period of seasonality
    - 7-day window to smooth weekly seasonality
    - Average out day level effects to produce the average week effect
"""

rolling_subs = usa_subscriptions.subs.rolling(
                #How many data point to average over
                window = 7,
                #Specify to average backwards
                center = False)

usa_subscriptions['rolling_subs'] = rolling_subs.mean()

usa_subscriptions.tail()

"""**Noisy data - Highest SKU purchased by date**

- Noisy Data: data with high variation over time

"""

high_sku_purchases = pd.read_csv(
                        'high_sku_purchases.csv',
                        parse_dates = True,
                        infer_datetime_format = True)

#Calculate the exp. avg. over our high sku purchase count
#Span: Window to apply weights over
exp_mean = high_sku_purchases.purchases.ewm(span = 30)

#Find the weighted mean over this period
high_sku_purchases['exp_mean'] = exp_mean.mean()

# Compute 7_day_rev
daily_revenue['7_day_rev'] = daily_revenue.revenue.rolling(window=7,center=False).mean()

# Compute 28_day_rev
daily_revenue['28_day_rev'] = daily_revenue.revenue.rolling(window=28,center=False).mean()
    
# Compute 365_day_rev
daily_revenue['365_day_rev'] = daily_revenue.revenue.rolling(window=365,center=False).mean()
    
# Plot date, and revenue, along with the 3 rolling functions (in order)    
daily_revenue.plot(x='date', y=['revenue', '7_day_rev', '28_day_rev', '365_day_rev', ])
plt.show()