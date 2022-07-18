import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 8)

"""## Three-input models

### Make an input layer for home vs. away
Now you will make an improvement to the model you used in the previous chapter for regular season games. You know there is a well-documented home-team advantage in basketball, so you will add a new input to your model to capture this effect.

This model will have three inputs: team_id_1, team_id_2, and home. The team IDs will be integers that you look up in your team strength model from the previous chapter, and home will be a binary variable, 1 if team_1 is playing at home, 0 if they are not.
"""

games_season = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                           '18_Advanced Deep Learning with Keras/data/games_season.csv')
print(games_season.head())

games_tourney = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                            '18_Advanced Deep Learning with Keras/data/games_tourney.csv')
print(games_tourney.head())

from tensorflow.keras.layers import Embedding, Input, Flatten
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

team_strength_model.summary()

from tensorflow.keras.layers import Concatenate, Dense

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

"""### Make a model and compile it
Now that you've input and output layers for the 3-input model, wrap them up in a Keras model class, and then compile the model, so you can fit it to data and use it to make predictions on new data.


"""

# Make a model
model = Model([team_in_1, team_in_2, home_in], out)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

"""### Fit the model and evaluate
Now that you've defined a new model, fit it to the regular season basketball data.

Use the `model` you fit in the previous exercise (which was trained on the regular season data) and evaluate the model on data for tournament games (`games_tourney`).
"""

# Fit the model to the games_season dataset
model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],
          games_season['score_diff'],
          epochs=1, verbose=True, validation_split=0.1, batch_size=2048)

# Evaluate the model on the games_touney dataset
print(model.evaluate([games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']], 
                     games_tourney['score_diff'], verbose=False))

"""## Summarizing and plotting models

### Model summaries
In this exercise, you will take a closer look at the summary of one of your 3-input models available in your workspace as model. Note how many layers the model has, how many parameters it has, and how many of those parameters are trainable/non-trainable.
"""

model.summary()

"""### Plotting models
In addition to summarizing your model, you can also plot your model to get a more intuitive sense of it.
"""

from tensorflow.keras.utils import plot_model

# Plot model
plot_model(model, to_file='./datacamp_repo/ML_Scientist_Career_Track/'
                          '18_Advanced Deep Learning with Keras/data/team_strength_model.png')

# Display the image
data = plt.imread('./datacamp_repo/ML_Scientist_Career_Track/'
                  '18_Advanced Deep Learning with Keras/data/team_strength_model.png')
plt.imshow(data)

"""## Stacking models

### Add the model predictions to the tournament data
In lesson 1 of this chapter, you used the regular season model to make predictions on the tournament dataset, and got pretty good results! Try to improve your predictions for the tournament by modeling it specifically.

You'll use the prediction from the regular season model as an input to the tournament model. This is a form of "model stacking."

To start, take the regular season model from the previous lesson, and predict on the tournament data. Add this prediction to the tournament data as a new column.
"""

# Predict
games_tourney['pred'] = model.predict([games_tourney['team_1'], 
                                       games_tourney['team_2'], 
                                       games_tourney['home']])

"""### Create an input layer with multiple columns
In this exercise, you will look at a different way to create models with multiple inputs. This method only works for purely numeric data, but its a much simpler approach to making multi-variate neural networks.

Now you have three numeric columns in the tournament dataset: `'seed_diff'`, `'home'`, and `'pred'`. In this exercise, you will create a neural network that uses a single input layer to process all three of these numeric inputs.

This model should have a single output to predict the tournament game score difference.
"""

# Create an input layer with 3 columns
input_tensor = Input(shape=(3, ))

# Pass it to a Dense layer with 1 unit
output_tensor = Dense(1)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

"""### Fit the model
Now that you've enriched the tournament dataset and built a model to make use of the new data, fit that model to the tournament data.

Note that this `model` has only one input layer that is capable of handling all 3 inputs, so it's inputs and outputs do not need to be a list.

Tournament games are split into a training set and a test set. The tournament games before 2010 are in the training set, and the ones after 2010 are in the test set.
"""

from sklearn.model_selection import train_test_split

games_tourney_train = games_tourney[games_tourney['season'] <= 2010]
games_tourney_test = games_tourney[games_tourney['season'] > 2010]

# Fit the model
model.fit(games_tourney_train[['home', 'seed_diff', 'pred']],
          games_tourney_train['score_diff'],
          epochs=1,
          verbose=True)

"""## Evaluate the model
Now that you've fit your model to the tournament training data, evaluate it on the tournament test data. Recall that the tournament test data contains games from after 2010.


"""

# Evaluate the model on the games_tourney_test dataset
print(model.evaluate(games_tourney_test[['home', 'seed_diff', 'pred']],
                     games_tourney_test['score_diff'],
                     verbose=True))

