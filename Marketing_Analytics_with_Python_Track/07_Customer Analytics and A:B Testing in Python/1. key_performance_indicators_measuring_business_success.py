import pandas as pd

customer_data = pd.read_csv("./datasets/user_demographics_v1.csv")
app_purchases = pd.read_csv("./datasets/purchase_data_v1.csv")


print(customer_data.columns)
print(app_purchases.columns)

"""**Merging Mechanics**"""

uid_combined_data = app_purchases.merge(
                    #right dataframe
                    customer_data,
                    #join type
                    how = 'inner',
                    #columns to match
                    on = ['uid'])

print(uid_combined_data.head())

uid_date_combined_data = app_purchases.merge(customer_data, on=['uid', 'date'], how = 'inner')

print(uid_date_combined_data.head())

print(customer_data.columns)
print(app_purchases.columns)

"""**Group by**"""

sub_data_grp = sub_data_demo.groupby(by=['country', 'device'], axis = 0, as_index = False)

sub_data_grp.price.mean()
sub_data_grp.price.agg('mean')
sub_data_grp.price.agg(['mean', 'median'])
sub_data_grp.agg({'price': ['mean', 'min', 'max'],
                  'age': ['mean', 'min', 'max']})
sub_data_grp.agg({'price': [custom_function]})

purchase_price_mean = uid_combined_data.price.agg('mean')

print(purchase_price_mean)

purchase_price_summary = uid_combined_data.price.agg(['mean', 'median'])

print(purchase_price_summary)

purchase_summary = uid_combined_data.agg({'price': ['mean', 'median'], 'age': ['mean', 'median']})

print(purchase_summary)

purchase_data = uid_combined_data

grouped_purchase_data = purchase_data.groupby(by = ['device', 'gender'])

purchase_summary = grouped_purchase_data.agg({'price':['mean', 'median', 'std']})

print(purchase_summary)

from pandas import Timestamp
from datetime import timedelta
current_date = Timestamp(2018,3,17)

max_purchase_date = current_date - timedelta(days = 28)

purchase_data['reg_date'] = pd.to_datetime(purchase_data['reg_date'])

purchase_data_filt = purchase_data[purchase_data['reg_date'] < max_purchase_date]

purchase_data_filt = purchase_data_filt[(purchase_data_filt.date <= purchase_data_filt.reg_date + timedelta(days = 28))]

print(purchase_data_filt.price.mean())