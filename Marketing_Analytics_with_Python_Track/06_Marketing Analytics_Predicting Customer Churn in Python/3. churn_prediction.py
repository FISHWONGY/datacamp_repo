import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

telco = pd.read_csv('./datasets/Churn.csv')
print(telco.head())

"""**First, we are going to remove categorical variables**"""

telco_cl_df = telco.drop(columns=['State', 'Area_Code', 'Phone'])

telco_cl_df.info()

"""**Then, we are going to Encode the Binary Features**"""

telco_cl_df['Vmail_Plan'] = telco_cl_df['Vmail_Plan'].replace({'no': 0, 'yes': 1})
telco_cl_df['Churn'] = telco_cl_df['Churn'].replace({'no': 0, 'yes': 1})
telco_cl_df['Intl_Plan'] = telco_cl_df['Intl_Plan'].replace({'no': 0, 'yes': 1})

print(telco_cl_df.head())

"""**Last, let's fit the scale so every feature is in the same scale**"""

telco_scaled = pd.DataFrame(StandardScaler().fit_transform(telco_cl_df))

print(telco_scaled.describe())

"""**Logistic Regression**

* In this example the code is just to ilustrate the process. The dataset in the course changed and hasn't been provided
"""

from sklearn.linear_model import LogisticRegression

#Instantiate the classifier
clf = LogisticRegression()

#Fit the classifier
clf.fit(telco[features], telco['Churn'])

#predict the label of the new customer
print(clf.predict(new_customer))

"""**Decision Tree**"""

from sklearn.tree import DecisionTreeClassifier

# Instantiate the classifier
clf = DecisionTreeClassifier()

# Fit the classifier
clf.fit(telco[features], telco['Churn'])

# Predict the label of new_customer
print(clf.predict(new_customer))

"""## Evaluating Model Performance

- Accuracy
    \begin{equation*}
    \text{Accuracy} = \frac{\text{Total Number of Correct Predictions}}{\text{Total Number of Data Points}}
    \end{equation*}

**Training and Test Sets**
- Fit your classifier to the training set
- Make predictions using the test set
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(telco['data'], telco['target'], test_size = 0.2, random_state = 42)

svc = SVC()

svc.fit(X_train, y_train)

svc.predict(X_test)

svc.score(X_test, y_test)

"""**Random Forest**"""

from sklearn.ensamble import RandomForestClassifier

# Instantiate the classifier
clf = RandomForestClassifier()

# Fit to the training data
clf.fit(X_train, y_train)

# Compute accuracy
print(clf.score(X_test, y_test))

"""**Model Metrics**

Imbalanced classess
- more data points that belong to one class than to another

This problem can be solved with *upsampling* and *downsampling*. This will help to balance the classes and help to solve this problem.

    **Recall/Sensitivity**

\begin{equation*}
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}
\end{equation*}

- A model with high precision indicates:
    - Few false positives ("false alarms")
    - Not many non-churners were classified as churners

    **Recall/Sensitivity**

\begin{equation*}
\text{Recall/Sensitivity} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}
\end{equation*}

- A model with high recall indicates that it correctly classified most churners


**Confusion Matrix in scikit-learn**
"""

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision_score(y_test, y_pred)
recall_score(y_test, y_pred)

"""**Other model metrics**

    The Receiving Operating Characteristic Curve (ROC)
    
Allows to visualize the performance of the churn classifier

- Every prediction the classifier makes has an associated probability
- **Default probability** treshold in scikit-learn is **50%**
    - if the probability is **> 50%** the model will predict the data point as belonging to the **positive** class
    - if the probability is **< 50%** the model will predict the data point as belonging to the **negative** class
"""

from sklearn.metrics import roc_curve

fpr, tpr, tresholds = roc_curve(y_test, y_pred_prob)

fpr = [0.        , 0.        , 0.        , 0.00116959, 0.00467836,
       0.01520468, 0.02923977, 0.04912281, 0.09005848, 0.18479532,
       0.4       , 1.        ]

tpr = [0.        , 0.16551724, 0.33793103, 0.44137931, 0.55172414,
       0.63448276, 0.73103448, 0.7862069 , 0.82758621, 0.84827586,
       0.87586207, 1.        ]

import matplotlib.pyplot as plt

plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.plot([0,1], [0,1], "k--")

plt.show()

"""**Area Under the Curve (AUC)**

The AUC indicates how well our model is behaving. An AUC >> 0.5 is a good one.
"""

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_pred)

"""**Generating Probabilities in sklearn**"""

logreg.predict_proba(X_test)[:,1]

"""**F1 Score**

\begin{equation*}
\text{F1 Score} = 2 \frac{\text{Precision * Recall}}{\text{Precision + Recall}}
\end{equation*}

- A high F1 score is sign of well-performing model
"""

# Import f1_score
from sklearn.metrics import f1_score

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Print the F1 score
print(f1_score(y_test, y_pred))