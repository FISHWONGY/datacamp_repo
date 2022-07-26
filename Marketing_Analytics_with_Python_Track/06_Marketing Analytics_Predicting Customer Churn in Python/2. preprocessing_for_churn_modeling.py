import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

telco = pd.read_csv('./datasets/Churn.csv')
print(telco.head())

"""**Model Assumptions**
- Some assumptions that models make:
    - That features are normally distributed
    - That features are on the same scale

**Data types**
- Machine learning algorithms require numeric data types
    - Need to encode categorical variable as numeric

**Standarization**
- Centers the distribution around the mean
- Calculates the number of standard deviations away from the mean each point is

## Encoding Binary Features

First we are going to recast some features. Here we are going to assign the values *1* to *'yes'* and *0* to *'no'*
"""

telco['Vmail_Plan'] = telco['Vmail_Plan'].replace({'no': 0, 'yes': 1})
telco['Churn'] = telco['Churn'].replace({'no': 0, 'yes': 1})
telco['Intl_Plan'] = telco['Intl_Plan'].replace({'no': 0, 'yes': 1})

print(telco['Vmail_Plan'].head())
print(telco['Churn'].head())

"""## One hot encoding

We are going to use this technique to encode the state
"""

telco_state = pd.get_dummies(telco['State'])

print(telco_state.head())

"""## Feature Scaling
Let's fit the scale so every feature is in the same scale
"""

telco_to_scale = telco[['Intl_Calls', 'Night_Mins']]
telco_scaled = StandardScaler().fit_transform(telco_to_scale)
telco_scaled_df = pd.DataFrame(telco_scaled, columns = ['Intl_Calls', 'Night_Mins'])

print(telco_scaled_df.describe())

"""## Feature Selection and Engineering

**Dropping unnecessary features**
- Unique identifiers
    - Phone numbers
    - Social security numbers
    - Account numbers

**Dropping correlated features**
- Highly correlated features can be dropped
- They provide no additional information to the model

**Feature engineering**
- Creating new features to help impove model performance
- Should consult with business and subject matter experts
"""

telco = telco.drop(['Area_Code', 'Phone'], axis = 1)
print(telco.columns)

telco['Avg_Night_Calls'] = telco['Night_Mins'] / telco['Night_Calls']

print(telco['Avg_Night_Calls'].head())