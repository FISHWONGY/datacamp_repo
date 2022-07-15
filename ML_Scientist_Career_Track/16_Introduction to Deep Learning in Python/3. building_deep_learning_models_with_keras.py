import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 8)

"""### Creating a keras model
- Model building steps
    - Specify Architecture
    - Compile
    - Fit
    - Predict

> Note: In the lecture, keras framework was used. But in this page, keras with tensorflow (`tf.keras`) will be used.

### Understanding your data
You will soon start building models in Keras to predict wages based on various professional and demographic factors. Before you start building a model, it's good to understand your data by performing some exploratory analysis.

The data is pre-loaded into a pandas DataFrame called `df`. Use the `.head()` and `.describe()` methods.

The target variable you'll be predicting is `wage_per_hour`. Some of the predictor variables are binary indicators, where a value of 1 represents True, and 0 represents False.
"""

df = pd.read_csv('./dataset/hourly_wages.csv')
df.head()

df.describe()

"""### Specifying a model
Now you'll get to work with your first model in Keras, and will immediately be able to run more complex neural network models on larger datasets compared to the first two chapters.

To start, you'll take the skeleton of a neural network and add a hidden layer and an output layer. You'll then fit that model and see Keras do the optimization so your model continually gets better.

As a start, you'll predict workers wages based on characteristics like their industry, education and level of experience. You can find the dataset in a pandas dataframe called `df`. For convenience, everything in `df` except for the target has been converted to a NumPy matrix called `predictors`. The target, `wage_per_hour`, is available as a NumPy matrix called `target`.
"""

import tensorflow as tf

predictors = df.iloc[:, 1:].to_numpy()
target = df.iloc[:, 0].to_numpy()

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = tf.keras.Sequential()

# Add the first layer
model.add(tf.keras.layers.Dense(50, activation='relu', input_shape=(n_cols, )))

# Add the second layer
model.add(tf.keras.layers.Dense(32, activation='relu'))

# Add the output layer
model.add(tf.keras.layers.Dense(1))

"""## Compiling and fitting a model
- Why you need to compile your model
    - Specify the optimizer
        - Many options and mathematically complex
        - "Adam" is usually a good choice
    - Loss function
        - "mean_squared_error"
- Fitting a model
    - Applying backpropagation and gradient descent with your data to update the weights
    - Scaling data before fitting can ease optimization

### Compiling the model
You're now going to compile the model you specified earlier. To compile the model, you need to specify the optimizer and loss function to use. You can read more about 'adam' optimizer as well as other keras optimizers [here](https://keras.io/optimizers/#adam), and if you are really curious to learn more, you can read the [original paper](https://arxiv.org/abs/1412.6980v8) that introduced the Adam optimizer.

In this exercise, you'll use the Adam optimizer and the mean squared error loss function. Go for it!
"""

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

"""### Fitting the model
You're at the most fun part. You'll now fit the model. Recall that the data to be used as predictive features is loaded in a NumPy matrix called `predictors` and the data to be predicted is stored in a NumPy matrix called `target`. Your model is pre-written and it has been compiled with the code from the previous exercise.
"""

# Fit the model
model.fit(predictors, target, epochs=10);

"""## Classification models
- Classification
    - `categorical_crossentropy` loss function
    - Similar to log loss: Lower is better
    - Add `metrics=['accuracy']` to compile step for easy-to-understand diagnostics
    - Output layers has separate node for each possible outcome, and uses `softmax` activation

### Understanding your classification data
Now you will start modeling with a new dataset for a classification problem. This data includes information about passengers on the Titanic. You will use predictors such as `age`, `fare` and where each passenger embarked from to predict who will survive. This data is from [a tutorial on data science competitions](https://www.kaggle.com/c/titanic). Look [here](https://www.kaggle.com/c/titanic/data) for descriptions of the features.

It's smart to review the maximum and minimum values of each variable to ensure the data isn't misformatted or corrupted. What was the maximum age of passengers on the Titanic?
"""

df = pd.read_csv('./dataset/titanic_all_numeric.csv')
df.head()

df.describe()

"""### Last steps in classification models
You'll now create a classification model using the titanic dataset, which has been pre-loaded into a DataFrame called `df`. You'll take information about the passengers and predict which ones survived.

The predictive variables are stored in a NumPy array `predictors`. The target to predict is in `df.survived`, though you'll have to manipulate it for keras. The number of predictive features is stored in `n_cols`.

Here, you'll use the `'sgd'` optimizer, which stands for [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). 
"""

predictors = df.iloc[:, 1:].astype(np.float32).to_numpy()
target = df.survived.astype(np.float32).to_numpy()
n_cols = predictors.shape[1]

from tensorflow.keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(target)

# Set up the model
model = tf.keras.Sequential()

# Add the first layer
model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(n_cols, )))

# Add the second layer
model.add(tf.keras.layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target, epochs=10);

"""## Using models
- Using models
    - Save
    - Load
    - Make predictions

### Making predictions
The trained network from your previous coding exercise is now stored as `model`. New data to make predictions is stored in a NumPy array as `pred_data`. Use model to make predictions on your new data.

In this exercise, your predictions will be probabilities, which is the most common way for data scientists to communicate their predictions to colleagues.
"""

pred_data = pd.read_csv('./dataset/titanic_pred.csv').astype(np.float32).to_numpy()

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:, 1]

# Print predicted_prob_true
print(predicted_prob_true)