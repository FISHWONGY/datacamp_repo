import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

# Review of pipelines using sklearn
'''
Pipeline review
 - Takes a list of 2-tuples (name, pipeline_step) as input
 - Tuples can contain any arbitrary scikit-learn compatible estimator or transformer object
 - Pipeline implements fit/predict methods
 - Can be used as input estimator into grid/randomized search and cross_val_score methods
'''
df = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                 '05_Extreme Gradient Boosting with XGBoost/data/ames_unprocessed_data.csv')

# Encoding categorical columns I - LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == 'object')

# Get list of categorical columns names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncode to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())


###
# Encoding categorical columns II - OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

df = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                 '05_Extreme Gradient Boosting with XGBoost/data/ames_unprocessed_data.csv')

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == 'object')

# Get list of categorical columns names
categorical_columns = df.columns[categorical_mask].tolist()

# Generate unique list of each categorical columns
unique_list = [df[c].unique().tolist() for c in categorical_columns]

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categories=unique_list)

# Create preprocess object for onehotencoding
preprocess = make_column_transformer(
    (ohe, categorical_columns),
    ('passthrough', categorical_mask[~categorical_mask].index.tolist())
)

# apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = preprocess.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape fo the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)


###
# Encoding categorical columns III: DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
df_dict = df.to_dict("records")

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded2 = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded2[:5, :])

# Print the vocabulary
print(dv.vocabulary_)


###
# Preprocessing within a pipeline
df = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                 '05_Extreme Gradient Boosting with XGBoost/data/ames_unprocessed_data.csv')
X, y = df.iloc[:, :-1], df.iloc[:, -1]

from sklearn.pipeline import Pipeline

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [('ohe_onestep', DictVectorizer(sparse=False)),
         ('xgb_model', xgb.XGBRegressor())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict("records"), y)


###
# Incorporating XGBoost into pipelines
'''
Additional components introduced for pipelines
sklearn_pandas:
   - DataFrameMapper - Interoperability between pandas and scikit-learn
   - CategoricalImputer - Allow for imputation of categorical variables before conversion to integers

sklearn.preprocessing:
   - Imputer - Native imputation of numerical columns in scikit-learn

sklearn.pipeline:
   - FeatureUnion - combine multiple pipelines of features into a single pipeline of features
'''

# Cross-validating your XGBoost model
df = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                 '05_Extreme Gradient Boosting with XGBoost/data/ames_unprocessed_data.csv')
X, y = df.iloc[:, :-1], df.iloc[:, -1]

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective='reg:squarederror'))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict('records'), y,
                                   scoring='neg_mean_squared_error', cv=10)

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))


###
# Kidney disease case study I - Categorical Imputer
# X - DF; y - array
X = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                '05_Extreme Gradient Boosting with XGBoost/data/chronic_kidney_X.csv')
y = pd.read_csv('/Volumes/My Passport for Mac/Python/Online course/datacamp/ML Scientist Career Track/'
                '05_Extreme Gradient Boosting with XGBoost/data/chronic_kidney_y.csv').to_numpy().ravel()

from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.impute import SimpleImputer

# Check number of nulls in each feature columns
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
    [([numeric_feature], SimpleImputer(strategy='median'))
     for numeric_feature in non_categorical_columns],
    input_df=True,
    df_out=True
)

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
    [(category_feature, CategoricalImputer())
     for category_feature in categorical_columns],
    input_df=True,
    df_out=True
)


###
# Kidney disease case study II - Feature Union
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
    ("num_mapper", numeric_imputation_mapper),
    ("cat_mapper", categorical_imputation_mapper)
])


###
# Kidney disease case study III - Full pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Define Dictifier class to turn df into dictionary as part of pipeline
class Dictifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if type(X) == pd.core.frame.DataFrame:
            return X.to_dict("records")
        else:
            return pd.DataFrame(X).to_dict("records")


# Create full pipeline
pipeline = Pipeline([
    ("featureunion", numeric_categorical_union),
    ("dictifier", Dictifier()),
    ("vectorizer", DictVectorizer(sort=False)),
    ("clf", xgb.XGBClassifier(max_depth=3))
])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))


###
# Tuning XGBoost hyperparameters
# Bringing it all together
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05, 1, 0.05),
    'clf__max_depth': np.arange(3, 10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator=pipeline, param_distributions=gbm_param_grid,
                                        n_iter=2, scoring='roc_auc', cv=2, verbose=1)

# Fit the estimator
randomized_roc_auc.fit(X, y)

# Compute metrics
print('Score: ', randomized_roc_auc.best_score_)
print('Estimator: ', randomized_roc_auc.best_estimator_)
print("Best parameters found: ", randomized_roc_auc.best_params_)

# Final Thoughts
'''
Advanced Topic
 - Using XGBoost for ranking/recommandation problems (Netflix/Amazon problem)
 - Using more sophisticated hyperparamter tuning strategies for tuning XGBoost model (Bayesian Optimization)
 - Using XGBoost as part of an ensemble of other models for regression/classification
'''