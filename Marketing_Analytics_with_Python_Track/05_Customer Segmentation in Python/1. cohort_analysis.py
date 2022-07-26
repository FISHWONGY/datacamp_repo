import pandas as pd
import datetime as dt

online = pd.read_csv('./datasets/chapter_1/online.csv', parse_dates = ['InvoiceDate'])
print(online.info())

"""We are going to use for this excercise the *date*, *price* & *customerID*

"""


def get_month(date):
    return dt.datetime(date.year, date.month, 1)

online['InvoiceMonth'] = online['InvoiceDate'].apply(get_month)

print(online.head())

grouping = online.groupby('CustomerID')['InvoiceMonth']
print(grouping.head())

online['CohortMonth'] = grouping.transform('min')
print(online.head())


def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day


invoice_year, invoice_month, _ = get_date_int(online, 'InvoiceMonth')
cohort_year, cohort_month, _ = get_date_int(online, 'CohortMonth')

years_diff = invoice_year - cohort_year
months_diff = invoice_month - cohort_month

online['CohortIndex'] = years_diff * 12 + months_diff + 1
print(online.head())

"""Count monthly active customers from each cohort"""

grouping = online.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)
cohort_data = cohort_data.reset_index()
cohort_counts = cohort_data.pivot(index = 'CohortMonth',
                                  columns = 'CohortIndex',
                                  values = 'CustomerID')
print(online.head())

"""## Calculate Cohort Metrics

Store the first column of the cohort as *cohort_size*
"""

cohort_sizes = cohort_counts.iloc[:,0]

"""Divide all values in the *cohort_counts* table by *cohort_sizes*"""

retention = cohort_counts.divide(cohort_sizes, axis = 0)
print(retention)

"""**Other Metrics**


"""

#Average
grouping = online.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['Quantity'].mean()
cohort_data = cohort_data.reset_index()
average_quantity = cohort_data.pivot(index='CohortMonth',
                                     columns='CohortIndex',
                                     values='Quantity')
average_quantity.round(1)

"""## Cohort Analysis Visualization"""

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.title('Retention rates')

sns.heatmap(data=retention,
            annot=True,
            fmt='.0%',
            vmin=0.0,
            vmax=0.5,
            cmap='BuGn')
plt.show()

plt.figure(figsize=(10, 8))
plt.title('Average Spend by Month Cohorts')
sns.heatmap(data=average_quantity,
            annot=True,
            cmap='Blues')
plt.show()
