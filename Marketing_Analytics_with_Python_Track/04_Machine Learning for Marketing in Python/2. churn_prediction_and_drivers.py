import pandas as pd

telcom = pd.read_csv('./datasets/telco.csv')
telcom.head()

set(telcom['Churn'])

telcom.groupby(['Churn']).size() / telcom.shape[0] * 100

"""**Excercie 1**

```python
    # Print the unique Churn values
    print(set(telcom['Churn']))

    # Calculate the ratio size of each churn group
    telcom.groupby(['Churn']).size() / telcom.shape[0] * 100

    # Import the function for splitting data to train and test
    from sklearn.model_selection import train_test_split

    # Split the data into train and test
    train, test = train_test_split(telcom, test_size = .25)
```

**Excercie 2**

```python
    # Store column names from `telcom` excluding target variable and customer ID
    cols = [col for col in telcom.columns if col not in custid + target]

    # Extract training features
    train_X = train[cols]

    # Extract training target
    train_Y = train[target]

    # Extract testing features
    test_X = test[cols]

    # Extract testing target
    test_Y = test[target]
```

## Predict Churn with logistic regresion

**Introduction to Logistic Regression**
- Statisticals classification model for binary responses
- Models log-odds of the probability of the target
- Assumes linear relationship between log-odds target and predictors
- Return coefficientos and prediction probability

\begin{equation*}
log_b\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \beta_2 x_2
\end{equation*}

**Modeling Steps**
- Spit data to training and testing
- Initialize the model
- Fit the model on the training
- Predict values on the testing data
- Measre the performance

**Model performance merics**
Key Metrics:
- Acuracy - The % of correctly predicted labels (Churn and no churn)
- Precision  - The % of total model's positive class predictions that were correctly classified
- Recall - The % of total positive class samples that were correctly classified

**Regularization**
- Introduces penalty coefficient in the model building phase
- Addresss over-fitting (when patterns re "memorized by the model")
- Some regularization techniques also perform feature selection e.g. L1
- Makes the model more generalizable to unseen samples

```python
    from sklearn.linear_model import LogistcRegression
    logreg = LogisticRegression(penalty = 'l1', C = 0.1, sover = 'liblinear')
    logreg.fit(train_X, train_Y)
```

**Tuning L1 regularization**
```python
    C = [1, .5, .25, .1, .05, .025, .01, .005, .0025]
    l1_metrics = np.zeros((len(C), 5))
    l1_metrics[:,0] = C
    
    for index in range(0, len(C)):
        logreg = LogisticRegrssion(penaly = 'l1', C = C[index], solver 'liblinear')
        logreg.fit(train_X, train_Y)
        pred_test_Y = loreg.predict(test_X)
        
        l1_metrics[index, 1] = np.count_nonzero(logreg-coef_)
        l1_metrics[index, 2] = accuracy_score(test_Y, pred_test_Y)
        l1_metrics[index, 3] = precision_score(test_Y, pred_test_Y)
        l1_metrics[index, 4] = recall_score(test_Y, pred_test_Y)
    col_names = ['C', 'Non-Zero Coeffs', 'Accuracy', 'Precision', 'Recall']
    print(pd.DataFrame(l1_metrics, columns = col_names)
```

**Excercise 1**
```python
    # Fit logistic regression on training data
    logreg.fit(train_X, train_Y)

    # Predict churn labels on testing data
    pred_test_Y = logreg.predict(test_X)

    # Calculate accuracy score on testing data
    test_accuracy = accuracy_score(test_Y, pred_test_Y)

    # Print test accuracy score rounded to 4 decimals
    print('Test accuracy:', round(test_accuracy, 4))
```

**Excercise 2**
```python
    # Initialize logistic regression instance 
    logreg = LogisticRegression(penalty='l1', C=0.025, solver='liblinear')

    # Fit the model on training data
    logreg.fit(train_X, train_Y)

    # Predict churn values on test data
    pred_test_Y = logreg.predict(test_X)

    # Print the accuracy score on test data
    print('Test accuracy:', round(accuracy_score(test_Y, pred_test_Y), 4))
```

**Excercise 3**
```python
    # Run a for loop over the range of C list length
    for index in range(0, len(C)):
      # Initialize and fit Logistic Regression with the C candidate
      logreg = LogisticRegression(penalty='l1', C=C[index], solver='liblinear')
      logreg.fit(train_X, train_Y)
      # Predict churn on the testing data
      pred_test_Y = logreg.predict(test_X)
      # Create non-zero count and recall score columns
      l1_metrics[index,1] = np.count_nonzero(logreg.coef_)
      l1_metrics[index,2] = recall_score(test_Y, pred_test_Y)

    # Name the columns and print the array as pandas DataFrame
    col_names = ['C','Non-Zero Coeffs','Recall']
    print(pd.DataFrame(l1_metrics, columns=col_names))
```

## Predict Churn with decision trees

```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    
    myTree = DecisionTreeClassifier()
    treemodel = mytree.fit(train_X, train_Y)
    
    #Measuring Model Accuracy
    pred_train_Y = mytree.predict(train_X)
    pred_test_Y = mytree.predict(test_X)
    
    train_accuracy = accuracy_score(train_Y, pred_train_Y)
    test_accuracy = accuracy_score(test_Y, pred_test_Y)
```

**Tree depth parameter tuning**
```python
    depth_list = list(range(2, 15))
    depth_tuning = np.zeros((len(depth_list), 4))
    depth_tuning[:, 0] = depth_list #sets the Depth Values
    
    for index in range(len(depth_list)):
        mytree = DecisionTreeClassifier(max_depth = depth_list[index])
        mytree.fit(train_X, train_Y)
        pred_test_Y = mytree.predict(test_X)
        
        depth_tuning[index, 1] = accuracy_score(test_Y, pred_test_Y)
        depth_tuning[index, 2] = precision_score(test_Y, pred_test_Y)
        depth_tuning[index, 3] = recall_score(test_Y, pred_test_Y)
    
    col_names = ['Max_Depth', 'Accuracy', 'Precision', 'Recall']
    print(pd.DataFrame(depth_tunng, columns = col_names))
```

**Excercise 1**
```python
    # Initialize decision tree classifier
    mytree = tree.DecisionTreeClassifier()

    # Fit the decision tree on training data
    mytree.fit(train_X, train_Y)

    # Predict churn labels on testing data
    pred_test_Y = mytree.predict(test_X)

    # Calculate accuracy score on testing data
    test_accuracy = accuracy_score(test_Y, pred_test_Y)

    # Print test accuracy
    print('Test accuracy:', round(test_accuracy, 4))
```
**Excercise 2**
```python
    # Run a for loop over the range of depth list length
    for index in range(0, len(depth_list)):
      # Initialize and fit decision tree with the `max_depth` candidate
      mytree = DecisionTreeClassifier(max_depth=depth_list[index])
      mytree.fit(train_X, train_Y)
      # Predict churn on the testing data
      pred_test_Y = mytree.predict(test_X)
      # Calculate the recall score 
      depth_tuning[index,1] = recall_score(test_Y, pred_test_Y)

    # Name the columns and print the array as pandas DataFrame
    col_names = ['Max_Depth','Recall']
    print(pd.DataFrame(depth_tuning, columns=col_names))
```

## Identify and Interpret churn drivers

**Plotting decision tree rules**

```python
    from sklearn import tree
    import graphviz
    
    exported = tree.export_graphviz(
        decision_tree = mytree,
        outfile = None,
        featue_names = cols,
        precision = 1,
        class_names = ['Not churn', 'Churn']
        filled = True
    )
    
    graph = graphviz.Source(exported)
    display(graph)
    
```

**Extracting logictic regression coefficients**

```python
    #Beta coeficients for each variable
    logreg.coef_
    
    #Transform the logictic regression ceffiients
    coefficients = pd.concat([pd.DataFrame(train_X.columns),
                             pd.DataFrame(np.transpose(logit.coef_))],
                            axis = 1)
    coefficients.columns = ['Feature', 'Coefficient']
    
    coefficients['Exp_Coefficient'] = np.exp(coefficients['Coefficient'])
    
    #remove the 0 value coefficients
    coefficients= coefficients[coefficients['Coefficient' != 0]]
    
    #print sorted bye the largest Coeficients values
    print(coefficients.sort_values(by = ['Coefficient']))
    
```

**Excercise 1**
```python
    # Combine feature names and coefficients into pandas DataFrame
    feature_names = pd.DataFrame(train_X.columns, columns = ['Feature'])
    log_coef = pd.DataFrame(np.transpose(logreg.coef_), columns = ['Coefficient'])
    coefficients = pd.concat([feature_names, log_coef], axis = 1)

    # Calculate exponent of the logistic regression coefficients
    coefficients['Exp_Coefficient'] = np.exp(coefficients['Coefficient'])

    # Remove coefficients that are equal to zero
    coefficients = coefficients[coefficients['Coefficient']!=0]

    # Print the values sorted by the exponent coefficient
    print(coefficients.sort_values(by=['Exp_Coefficient']))
```

**Excercise 2**
```python
    # Export graphviz object from the trained decision tree 
    exported = tree.export_graphviz(decision_tree=mytree, 
                # Assign feature names
                out_file=None, feature_names=train_X.columns, 
                # Set precision to 1 and add class names
                precision=1, class_names=['Not churn','Churn'], filled = True)

    # Call the Source function and pass the exported graphviz object
    graph = graphviz.Source(exported)
```
"""