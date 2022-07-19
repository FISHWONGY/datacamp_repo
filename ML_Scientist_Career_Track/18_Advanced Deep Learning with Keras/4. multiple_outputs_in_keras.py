import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 8)

"""## Two-output models

### Simple two-output model
In this exercise, you will use the tournament data to build one model that makes two predictions: the scores of both teams in a given game. Your inputs will be the seed difference of the two teams, as well as the predicted score difference from the model you built in chapter 3.

The output from your model will be the predicted score for team 1 as well as team 2. This is called "multiple target regression": one model making more than one prediction.
"""

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define the input
input_tensor = Input(shape=(2, ))

# Define the output
output_tensor = Dense(2)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

"""### Fit a model with two outputs
Now that you've defined your 2-output model, fit it to the tournament data. I've split the data into `games_tourney_train` and `games_tourney_test`, so use the training set to fit for now.

This model will use the pre-tournament seeds, as well as your pre-tournament predictions from the regular season model you built previously in this course.

As a reminder, this model will predict the scores of both teams.
"""

games_tourney = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                            '18_Advanced Deep Learning with Keras/data/games_tourney.csv')
print(games_tourney.head())

games_season = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                           '18_Advanced Deep Learning with Keras/data/games_season.csv')
print(games_season.head())

from tensorflow.keras.layers import Embedding, Input, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model

# Count the unique number of teams
n_teams = np.unique(games_season['team_1']).shape[0]

# Create an embedding layer
team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')

# Create an input layer for the team ID
teamid_in = Input(shape=(1, ))

# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# Combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')

# Create an Input for each team
team_in_1 = Input(shape=(1, ), name='Team-1-In')
team_in_2 = Input(shape=(1, ), name='Team-2-In')

# Create an input for home vs away
home_in = Input(shape=(1, ), name='Home-In')

# Lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# Combine the team strengths with the home input using a Concatenate layer, 
# then add a Dense layer

out = Concatenate()([team_1_strength, team_2_strength, home_in])
out = Dense(1)(out)

# Make a model
p_model = Model([team_in_1, team_in_2, home_in], out)

# Compile the model
p_model.compile(optimizer='adam', loss='mean_absolute_error')

# Fit the model to the games_season dataset
p_model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],
            games_season['score_diff'],
            epochs=1, verbose=True, validation_split=0.1, batch_size=2048)

games_tourney['pred'] = p_model.predict([games_tourney['team_1'], 
                                        games_tourney['team_2'],
                                        games_tourney['home']])

games_tourney_train = games_tourney[games_tourney['season'] <= 2010]
games_tourney_test = games_tourney[games_tourney['season'] > 2010]

# Fit the model
model.fit(games_tourney_train[['seed_diff', 'pred']], 
          games_tourney_train[['score_1', 'score_2']],
          verbose=False,
          epochs=10000,
          batch_size=256)

"""### Inspect the model (I)
Now that you've fit your model, let's take a look at it. You can use the `.get_weights()` method to inspect your model's weights.

The input layer will have 4 weights: 2 for each input times 2 for each output.

The output layer will have 2 weights, one for each output.
"""

# Print the model's weight
print(model.get_weights())

# Print the column means of the training data
print(games_tourney_train.mean())

"""### Evaluate the model
Now that you've fit your model and inspected it's weights to make sure it makes sense, evaluate it on the tournament test set to see how well it performs on new data.


"""

# Evaluate the model on the tournament test data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']], 
                     games_tourney_test[['score_1', 'score_2']],
                     verbose=False))

"""## Single model for classification and regression

### Classification and regression in one model
Now you will create a different kind of 2-output model. This time, you will predict the score difference, instead of both team's scores and then you will predict the probability that team 1 won the game. This is a pretty cool model: it is going to do both classification and regression!

In this model, turn off the bias, or intercept for each layer. Your inputs (seed difference and predicted score difference) have a mean of very close to zero, and your outputs both have means that are close to zero, so your model shouldn't need the bias term to fit the data well.
"""

# Create an input layer with 2 columns
input_tensor = Input(shape=(2, ))

# Create the first output
output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor)

# Create the second output(use the first output as input here)
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False)(output_tensor_1)

# Create a model with 2 outputs
model = Model(input_tensor, [output_tensor_1, output_tensor_2])

"""### Compile and fit the model
Now that you have a model with 2 outputs, compile it with 2 loss functions: mean absolute error (MAE) for `'score_diff'` and binary cross-entropy (also known as logloss) for `'won'`. Then fit the model with `'seed_diff'` and `'pred'` as inputs. For outputs, predict `'score_diff'` and `'won'`.

This model can use the scores of the games to make sure that close games (small score diff) have lower win probabilities than blowouts (large score diff).

The regression problem is easier than the classification problem because MAE punishes the model less for a loss due to random chance. For example, if `score_diff` is -1 and `won` is 0, that means `team_1` had some bad luck and lost by a single free throw. The data for the easy problem helps the model find a solution to the hard problem.
"""

from tensorflow.keras.optimizers import Adam

# Compile the model with 2 losses and the Adam optimizer with a higher learning rate
model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer=Adam(lr=0.01))

# Fit the model to the tournament training data, with 2 inputs and 2 outputs
model.fit(games_tourney_train[['seed_diff', 'pred']],
          [games_tourney_train[['score_diff']], games_tourney_train[['won']]],
          epochs=20,
          verbose=True,
          batch_size=16384)

"""### Inspect the model (II)
Now you should take a look at the weights for this model. In particular, note the last weight of the model. This weight converts the predicted score difference to a predicted win probability. If you multiply the predicted score difference by the last weight of the model and then apply the sigmoid function, you get the win probability of the game.
"""

# Print the model weights
print(model.get_weights())

# Print the training data means
print(games_tourney_train.mean())

from scipy.special import expit as sigmoid

# Weight from the model
weight = 0.14

# Print the approximate win probability predicted close game
print(sigmoid(1 * weight))

# Print the approximate win probability predicted blowout game
print(sigmoid(10 * weight))

"""### Evaluate on new data with two metrics
Now that you've fit your model and inspected its weights to make sure they make sense, evaluate your model on the tournament test set to see how well it does on new data.

Note that in this case, Keras will return 3 numbers: the first number will be the sum of both the loss functions, and then the next 2 numbers will be the loss functions you used when defining the model.
"""

# Evaluate the model on new data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']],
                     [games_tourney_test[['score_diff']], games_tourney_test[['won']]],
                     verbose=False))
