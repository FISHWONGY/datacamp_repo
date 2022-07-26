import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

cohort_counts = pd.read_csv('some.csv')

# Extract cohort sizes from the first column of cohort_counts
cohort_sizes = cohort_counts.iloc[:, 0]

# Calculate retention by dividing the counts with the cohort sizes
retention = cohort_counts.divide(cohort_sizes, axis=0)

# Calculate churn
churn = 1 - retention

# Print the retention table
print(retention)

# Calculate the mean retention rate
retention_rate = retention.iloc[:, 1:].mean().mean()

# Calculate the mean churn rate
churn_rate = churn.iloc[:, 1:].mean().mean()

# Print rounded retention and churn rates
print('Retention rate: {:.2f}; Churn rate: {:.2f}'.format(retention_rate, churn_rate))

# Calculting and projecting CLV

# Calculate monthly spend per customer
online = pd.read_csv('some.csv')
monthly_revenue = online.grouby(['CustomerID', 'InvoiceMonth'])['TotalSum'].sum().mean()

# Calculate average monthly spend
monthly_revenue = np.mean(monthly_revenue)

# Define Lifespan to 36 months
lifespan_months = 36

# Calculate basic CLV
clv_basic = monthly_revenue * lifespan_months

# Print basic CLV value
print('Average basic CLV is {:.1f} USD.'.format(clv_basic))


# Calculate average revenue per invoice
revenue_per_purchase = online.groupby(['InvoiceNo'])['TotalSum'].mean().mean()

# Calculate average number of uniques invoices per customer per month
freq = online.groupby(['CustomerId', 'InvoiceMonth'])['InvoiceNo'].nunique().mean()

# Define Lifespan to 36 months
lifespan_months = 36

# Calculate granular CLV
clv_granular = revenue_per_purchase * freq * lifespan_months

# Print basic CLV value
print('Average granular CLV is {:.1f} USD.'.format(clv_granular))


# Calculate monthly spend per customer
monthly_revenue = online.groupby(['CusotmerID', 'InvoiceMonth'])['TotalSum'].sum().mean()

# Calculate average monthly retention rate
retention_rate = retention.iloc[:, 1:].mean().mean()

# Calculate average monthly churn rate
churn_rate = 1 - retention_rate

clv_traditional = monthly_revenue * (retention_rate / churn_rate)

# Print traditional CLV and the retention rate values
print('Average traditional CLV is {:.1f} USD at {:.1f} % retention_rate.'.format(clv_tradicional, retention_rate * 100))


# DataPreparation for purchase prediction

# Define the snapshot date
NOW = dt.datetime(2011, 11, 1)

# Calculate recency by subtracting current date from the latest InvoiceDate
features = online_X.groupby('CustomerID').agg({
  'InvoiceDate': lambda x: (NOW - x.max()).days,
  # Calculate frequency by counting unique number of invoices
  'InvoiceNo': pd.Series.nunique,
  # Calculate monetary value by summing all spend values
  'TotalSum': np.sum,
  # Calculate average and total quantity
  'Quantity': ['mean', 'sum']}).reset_index()

# Rename the columns
features.columns = ['CustomerID', 'recency', 'frequency', 'monetary', 'quantity_avg', 'quantity_total']


# Build a pivot table counting invoices for each customer monthly
cust_month_tx = pd.pivot_table(data=online, values='InvoiceNo',
                               index=['CustomerID'], columns=['InvoiceMonth'],
                               aggfunc=pd.Series.nunique, fill_value=0)

# Store November 2011 data column name as a list
target = ['2011-11']

# Store target value as `Y`
Y = cust_month_tx[target]


# Store customer identifier column name as a list
custid = ['CustomerID']

# Select feature column names excluding customer identifier
cols = [col for col in features.columns if col not in custid]

# Extract the features as `X`
X = features[cols]

# Split data to training and testing
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=99)

## Predicting Customer Transactions
# Initialize linear regression instance
linreg = LinearRegression()

# Fit the model to training dataset
linreg.fit(train_X, train_Y)

# Predict the target variable for training data
train_pred_Y = linreg.predict(train_X)

# Predict the target variable for testing data
test_pred_Y = linreg.predict(test_X)


# Calculate root mean squared error on training data
rmse_train = np.sqrt(mean_squared_error(train_Y, train_pred_Y))

# Calculate mean absolute error on training data
mae_train = mean_absolute_error(train_Y, train_pred_Y)

# Calculate root mean squared error on testing data
rmse_test = np.sqrt(mean_squared_error(test_Y, test_pred_Y))

# Calculate mean absolute error on testing data
mae_test = mean_absolute_error(test_Y, test_pred_Y)

# Print the performance metrics
print('RMSE train: {}; RMSE test: {}\nMAE train: {}, MAE test: {}'.format(rmse_train, rmse_test, mae_train, mae_test))

# Import `statsmodels.api` module
import statsmodels.api as sm

# Initialize model instance on the training data
olsreg = sm.OLS(train_Y, train_X)

# Fit the model
olsreg = olsreg.fit()

# Print model summary
print(olsreg.summary())

