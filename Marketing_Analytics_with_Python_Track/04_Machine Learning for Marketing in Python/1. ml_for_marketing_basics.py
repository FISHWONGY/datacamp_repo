import pandas as pd

telco_raw = pd.read_csv('./datasets/telco.csv')
telco_raw.head()

telco_raw.dtypes

"""**Separate categorical and numerical columns**

Separate identifier andtarget variable names as lists
"""

custid = ['customerID']
target = ['Churn']

"""Separate categorical ad numeric column names as lists"""

# A value is defined as categorical if it has less than 10 unique values
categorical = telco_raw.nunique()[telco_raw.nunique() < 10].keys().tolist()

#Remove the target variable called Churn from the list so we don't do any transformation on it
categorical.remove(target[0])

#Finally, we store the remaning column names into a list called numerical
numerical = [ col for col in telco_raw.columns
              if col not in custid + target + categorical]

"""**One-hot encoding in categorical values**

"""

telco_raw = pd.get_dummies(data = telco_raw, columns = categorical, drop_first = True)

"""**Scaling numerical features**"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_numerical = scaler.fit_transform(telco_raw[numerical])

scaled_numerical = pd.DataFrame(scaled_numeical, columns = numerical)

"""**Bringing All Together**"""

telco_raw = telco_raw.drop(columns = numerical, axis = 1)
telco = telco_raw.merge(right = scaled_numerical,
                        how = 'left',
                        left_index = True,
                        right_index = True
                       )

"""**Supervised Learning Steps**
- Spit data to training and testing
- Initialize the model
- Fit the model on the training
- Predict values on the testing data
- Measre the performance
"""

from sklearn import tree
from sklearn.model_selection import tran_test_split
from sklearn.metrics import accuracy_score

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.25)

# Ensure training dataset has only 75% of original X data
print(train_X.shape[0] / X.shape[0])

# Ensure testing dataset has only 25% of original X data
print(test_X.shape[0] / X.shape[0])

mytree = tree.DecisionTreeClassifier(max_depth = 5)

treemodel = mytree.fit(train_X, train_Y)

pred_Y = treemodel.predict(test_X)

accuracy_score(test_Y, pred_Y)

"""**Unsupervised learning steps**
- Initialize the model
- Fit the model
- Assign cluster values
- Explore the results
"""

from sklearn.cluster import KMeans
import pandas as pd

kmeans = KMeans(n_clusters = 3)

kmeans.fit(data)

data.assign(Cluster = kmeans.labels_)

data.groupby('Cluster').mean()