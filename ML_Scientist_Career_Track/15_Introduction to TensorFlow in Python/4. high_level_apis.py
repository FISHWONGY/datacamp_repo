import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 8)
print(tf.__version__)

"""## Defining neural networks with Keras

### The sequential model in Keras
n chapter 3, we used components of the keras API in tensorflow to define a neural network, but we stopped short of using its full capabilities to streamline model definition and training. In this exercise, you will use the keras sequential model API to define a neural network that can be used to classify images of sign language letters. You will also use the `.summary()` method to print the model's architecture, including the shape and number of parameters associated with each layer.

Note that the images were reshaped from (28, 28) to (784,), so that they could be used as inputs to a dense layer.
"""

# Define a Keras sequential model
model = tf.keras.Sequential()

# Define the first dense layer
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(tf.keras.layers.Dense(8, activation='relu', ))

# Define the output layer
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Print the model architecture
print(model.summary())

"""Notice that we've defined a model, but we haven't compiled it. The compilation step in keras allows us to set the optimizer, loss function, and other useful training parameters in a single line of code. Furthermore, the `.summary()` method allows us to view the model's architecture.

### Compiling a sequential model
In this exercise, you will work towards classifying letters from the Sign Language MNIST dataset; however, you will adopt a different network architecture than what you used in the previous exercise. There will be fewer layers, but more nodes. You will also apply dropout to prevent overfitting. Finally, you will compile the model to use the adam optimizer and the `categorical_crossentropy` loss. You will also use a method in keras to summarize your model's architecture.
"""

model = tf.keras.Sequential()

# Define the first dense layer
model.add(tf.keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Apply dropout to the first layer's output
model.add(tf.keras.layers.Dropout(0.25))

# Define the output layer
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())

"""### Defining a multiple input model
In some cases, the sequential API will not be sufficiently flexible to accommodate your desired model architecture and you will need to use the functional API instead. If, for instance, you want to train two models with different architectures jointly, you will need to use the functional API to do this. In this exercise, we will see how to do this. We will also use the `.summary()` method to examine the joint model's architecture.
"""

m1_inputs = tf.keras.Input(shape=(784,))
m2_inputs = tf.keras.Input(shape=(784,))

# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = tf.keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = tf.keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = tf.keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = tf.keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = tf.keras.layers.add([m1_layer2, m2_layer2])
model = tf.keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())

"""Notice that the `.summary()` method yields a new column: `connected to`. This column tells you how layers connect to each other within the network. We can see that `dense_9`, for instance, is connected to the `input_2` layer. We can also see that the add layer, which merged the two models, connected to both `dense_10` and `dense_12`.

## Training and validation with Keras

### Training with Keras
In this exercise, we return to our sign language letter classification problem. We have 2000 images of four letters--A, B, C, and D--and we want to classify them with a high level of accuracy. We will complete all parts of the problem, including the model definition, compilation, and training.
"""

df = pd.read_csv('./dataset/slmnist.csv', header=None)
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

sign_language_features = (X - X.mean()) / (X.max() - X.min()).to_numpy()
sign_language_labels = pd.get_dummies(y).astype(np.float32).to_numpy()

# Define a sequential model
model = tf.keras.Sequential()

# Define a hidden layer
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(784, )))

# Define the output layer
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)

"""You probably noticed that your only measure of performance improvement was the value of the loss function in the training sample, which is not particularly informative.

### Metrics and validation with Keras
We trained a model to predict sign language letters in the previous exercise, but it is unclear how successful we were in doing so. In this exercise, we will try to improve upon the interpretability of our results. Since we did not use a validation split, we only observed performance improvements within the training set; however, it is unclear how much of that was due to overfitting. Furthermore, since we did not supply a metric, we only saw decreases in the loss function, which do not have any clear interpretation.
"""

# Define sequential model
model = tf.keras.Sequential()

# Define the first layer
model.add(tf.keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# Add activation function to classifier
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)

"""With the keras API, you only needed 14 lines of code to define, compile, train, and validate a model. You may have noticed that your model performed quite well. In just 10 epochs, we achieved a classification accuracy of over 90% in the validation sample!

### Overfitting detection
In this exercise, we'll work with a small subset of the examples from the original sign language letters dataset. A small sample, coupled with a heavily-parameterized model, will generally lead to overfitting. This means that your model will simply memorize the class of each example, rather than identifying features that generalize to many examples.

You will detect overfitting by checking whether the validation sample loss is substantially higher than the training sample loss and whether it increases with further training. With a small sample and a high learning rate, the model will struggle to converge on an optimum. You will set a low learning rate for the optimizer, which will make it easier to identify overfitting.
"""

# Define sequential model
model = tf.keras.Sequential()

# Define the first layer
model.add(tf.keras.layers.Dense(1024, activation='relu', input_shape=(784, )))

# Add activation function to classifier
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=50, validation_split=0.5)

"""### Evaluating models
Two models have been trained and are available: `large_model`, which has many parameters; and `small_model`, which has fewer parameters. Both models have been trained using `train_features` and `train_labels`, which are available to you. A separate test set, which consists of `test_features` and `test_labels`, is also available.

Your goal is to evaluate relative model performance and also determine whether either model exhibits signs of overfitting. You will do this by evaluating `large_model` and `small_model` on both the train and test sets. For each model, you can do this by applying the `.evaluate(x, y)` method to compute the loss for features `x` and labels `y`. You will then compare the four losses generated.
"""

small_model = tf.keras.Sequential()

small_model.add(tf.keras.layers.Dense(8, activation='relu', input_shape=(784,)))
small_model.add(tf.keras.layers.Dense(4, activation='softmax'))

small_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

large_model = tf.keras.Sequential()

large_model.add(tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(784,)))
large_model.add(tf.keras.layers.Dense(4, activation='softmax'))

large_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, 
                                                       beta_1=0.9, beta_2=0.999),
                   loss='categorical_crossentropy', metrics=['accuracy'])

from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(sign_language_features, 
                                                                            sign_language_labels,
                                                                            test_size=0.5)

small_model.fit(train_features, train_labels, epochs=30, verbose=False)
large_model.fit(train_features, train_labels, epochs=30, verbose=False)

# Evaluate the small model using the train data
small_train = small_model.evaluate(train_features, train_labels)

# Evaluate the small model using the test data
small_test = small_model.evaluate(test_features, test_labels)

# Evaluate the large model using the train data
large_train = large_model.evaluate(train_features, train_labels)

# Evalute the large model using the test data
large_test = large_model.evaluate(test_features, test_labels)

# Print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))

"""## Training models with the Estimators API
- Estimators API
![estimators](image/estimators.png)
    - High level submodule
    - Less flexible
    - Faster deployment
    - Many premade model
- Model specification and training
    1. Define feature columns
    2. Load and transform data
    3. Define an estimator
    4. Apply train operation

### Preparing to train with Estimators
For this exercise, we'll return to the King County housing transaction dataset from chapter 2. We will again develop and train a machine learning model to predict house prices; however, this time, we'll do it using the `estimator` API.

Rather than completing everything in one step, we'll break this procedure down into parts. We'll begin by defining the feature columns and loading the data. In the next exercise, we'll define and train a premade `estimator`.
"""

housing = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                      '15_Introduction to TensorFlow in Python/data/kc_house_data.csv')
housing.head()

# Define feature columns for bedrooms and bathrooms
bedrooms = tf.feature_column.numeric_column("bedrooms")
bathrooms = tf.feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]

def input_fn():
    # Define the labels
    labels = np.array(housing['price'])
    
    # Define the features
    features = {'bedrooms': np.array(housing['bedrooms']),
                'bathrooms': np.array(housing['bathrooms'])}
    
    return features, labels

"""### Defining Estimators
In the previous exercise, you defined a list of feature columns, `feature_list`, and a data input function, `input_fn()`. In this exercise, you will build on that work by defining an estimator that makes use of input data.
"""

# Define the model and set the number of steps
model = tf.estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2,2])
model.train(input_fn, steps=1)

# Define the model and set the number of steps
model = tf.estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)

