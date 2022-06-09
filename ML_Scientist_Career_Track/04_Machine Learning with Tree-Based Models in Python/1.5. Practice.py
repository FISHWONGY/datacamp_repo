import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

mpg = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                  '04_Machine Learning with Tree-Based Models in Python/data/auto.csv')

X = mpg.drop(['mpg', 'origin'], axis='columns')
y = mpg['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Using max_depth of 8 led to overfitting.
tree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)
tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

print("MSE train: {0:.4f}, test: {1:.4f}".format(mean_squared_error(y_train, y_train_pred),
                                                 mean_squared_error(y_test, y_test_pred)))

print("R^2 train: {0:.4f}, test: {1:.4f}".format(r2_score(y_train, y_train_pred),
                                                 r2_score(y_test, y_test_pred)))

# According to Decisoiin Tree
result = pd.DataFrame(tree.feature_importances_, mpg.drop(['mpg', 'origin'], axis='columns').columns)
result.columns = ['feature']
result.sort_values(by='feature', ascending=False).plot(kind='bar')

# Displ is the most important feature

# Wanted to see if I'm over fitting by plotting it out, turns out this only works for 1 X variable but not multiple
'''
sort_idx = X_train.flatten().argsort()
plt.figure(figsize=(10, 8))
plt.scatter(X_train[sort_idx], y_train[sort_idx])
plt.plot(X_train[sort_idx], tree.predict(X_train[sort_idx]), color='k')

plt.xlabel('IDK')
plt.ylabel('mpg')
'''


mpg = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                  '04_Machine Learning with Tree-Based Models in Python/data/auto.csv')

# Both X, y has to be an array
X = mpg[['displ']].values
y = mpg['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Using max_depth of 8 led to overfitting.
tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=0.13, random_state=3)
tree.fit(X_train, y_train)

sort_idx = X_train.flatten().argsort()
plt.figure(figsize=(10, 8))
plt.scatter(X_train[sort_idx], y_train[sort_idx])
plt.plot(X_train[sort_idx], tree.predict(X_train[sort_idx]), color='k')

plt.xlabel('displ')
plt.ylabel('mpg')