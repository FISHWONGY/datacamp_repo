import pandas as pd

d = {'CustomerID': [0,1,2,3,4,5,6,7],
     'Spend': [137,335,172,355,303,233,244,229]}

data = pd.DataFrame(d, columns = ['CustomerID', 'Spend'])

print(data.head())

spend_quartile = pd.qcut(data['Spend'], q=4, labels = range(1,5))

data['Spend_Quartile'] = spend_quartile

print(data.sort_values('Spend'))

data = pd.DataFrame(columns = ['CustomerID', 'Recency_Days'], data = [[0,37], [1,235], [2, 396], [3, 72], [4, 255], [5, 393], [6, 203], [7, 133]])

data

#Store labels from 4 to 1 in a decresing order
r_labels = list(range(4, 0, -1))

recency_quartiles = pd.qcut(data['Recency_Days'], q = 4, labels = r_labels)

data['Recency_Quartile'] = recency_quartiles

print(data.sort_values('Recency_Days'))

online = pd.read_csv('./datasets/chapter_2/online12M.csv', parse_dates = ['InvoiceDate'])
online['TotalSum'] = online['Quantity'] * online['UnitPrice']
online.head()

print('Min:{}; Max:{}'.format(min(online.InvoiceDate), max(online.InvoiceDate)))

import datetime
snapshot_date = max(online.InvoiceDate) + datetime.timedelta(days = 1)
snapshot_date

#Aggregate data on a customer level
datamart = online.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

datamart.rename(columns = {'InvoiceDate': 'Recency',
                           'InvoiceNo': 'Frequency',
                           'TotalSum': 'MonetaryValue'}, inplace = True)

datamart.head()

"""## Buildng RFM Segments


"""

# The recency is better when it's lower
r_labels = range(4, 0, -1)

r_quartiles = pd.qcut(datamart['Recency'], 4, labels = r_labels)
datamart = datamart.assign(R = r_quartiles.values)

datamart.head()

# The frequency and monetary values are better when they are higher
f_labels = range(1, 5)
m_labels = range(1, 5)

f_quartiles = pd.qcut(datamart['Frequency'], 4, labels = f_labels)
m_quartiles = pd.qcut(datamart['MonetaryValue'], 4, labels = m_labels)

datamart = datamart.assign(F = f_quartiles.values)
datamart = datamart.assign(M = m_quartiles.values)

datamart.head()

"""**Build RFM Segment and RFM Score**

- Concatenate RFM quartile values to RFM_Segment
- Sum RFM quartiles vales to RFM_Score
"""

def join_rfm(x): return str(x['R']) + str(x['F']) + str(x['M'])

datamart['RFM_Segment'] = datamart.apply(join_rfm, axis = 1)
datamart['RFM_Score'] =datamart[['R', 'F', 'M']].sum(axis = 1)

datamart

"""**Analyzing RFM Segments**"""

datamart.groupby('RFM_Segment').size().sort_values(ascending = False)[:10]

datamart[datamart['RFM_Segment'] == '144']

datamart.groupby('RFM_Score').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count', 'sum']
}).round(1)

"""**Grouping into named segments**

Use RFM scoreto group customers into *Gold*, *Silver*, *Bronze* segments

"""

def segment_me(df):
    if df['RFM_Score'] >= 9:
        return 'Gold'
    elif (df['RFM_Score'] >= 5) and (df['RFM_Score'] < 9):
        return 'Silver'
    else:
        return 'Bronze'

datamart['General_Segment'] = datamart.apply(segment_me, axis = 1)

datamart.groupby('General_Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
}).round(1)