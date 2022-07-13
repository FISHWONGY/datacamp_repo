import pyspark
import numpy as np
import pandas as pd

"""## What is logistic regression?
The model you'll be fitting in this chapter is called a logistic regression. This model is very similar to a linear regression, but instead of predicting a numeric variable, it predicts the probability (between 0 and 1) of an event.

To use this as a classification algorithm, all you have to do is assign a cutoff point to these probabilities. If the predicted probability is above the cutoff point, you classify that observation as a 'yes' (in this case, the flight being late), if it's below, you classify it as a 'no'!

You'll tune this model by testing different values for several hyperparameters. A hyperparameter is just a value in the model that's not estimated from the data, but rather is supplied by the user to maximize performance. For this course it's not necessary to understand the mathematics behind all of these values - what's important is that you'll try out a few different choices and pick the best one.

## Create the modeler
The `Estimator` you'll be using is a `LogisticRegression` from the `pyspark.ml.classification` submodule.
"""

from pyspark.sql import SparkSession

spark = (SparkSession
  .builder
  .appName("flights")
  .getOrCreate())

from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()

"""## Cross validation
In the next few exercises you'll be tuning your logistic regression model using a procedure called k-fold cross validation. This is a method of estimating the model's performance on unseen data (like your `test` DataFrame).

It works by splitting the training data into a few different partitions. The exact number is up to you, but in this course you'll be using PySpark's default value of three. Once the data is split up, one of the partitions is set aside, and the model is fit to the others. Then the error is measured against the held out partition. This is repeated for each of the partitions, so that every block of data is held out and used as a test set exactly once. Then the error on each of the partitions is averaged. This is called the cross validation error of the model, and is a good estimate of the actual error on the held out data.

You'll be using cross validation to choose the hyperparameters by creating a grid of the possible pairs of values for the two hyperparameters, `elasticNetParam` and `regParam`, and using the cross validation error to compare all the different models so you can choose the best one!

## Create the evaluator
The first thing you need when doing cross validation for model selection is a way to compare different models. Luckily, the `pyspark.ml.evaluation` submodule has classes for evaluating different kinds of models. Your model is a binary classification model, so you'll be using the `BinaryClassificationEvaluator` from the `pyspark.ml.evaluation` module.

This evaluator calculates the area under the ROC. This is a metric that combines the two kinds of errors a binary classifier can make (false positives and false negatives) into a simple number. You'll learn more about this towards the end of the chapter!
"""

import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')

"""## Make a grid
Next, you need to create a grid of values to search over when looking for the optimal hyperparameters. The submodule `pyspark.ml.tuning` includes a class called `ParamGridBuilder` that does just that (maybe you're starting to notice a pattern here; PySpark has a submodule for just about everything!).

You'll need to use the `.addGrid()` and `.build()` methods to create a grid that you can use for cross validation. The `.addGrid()` method takes a model parameter (an attribute of the model `Estimator`, `lr`, that you created a few exercises ago) and a list of values that you want to try. The `.build()` method takes no arguments, it just returns the grid that you'll use later.
"""

import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()

"""## Make the validator
The submodule `pyspark.ml.tuning` also has a class called `CrossValidator` for performing cross validation. This `Estimator` takes the modeler you want to fit, the grid of hyperparameters you created, and the evaluator you want to use to compare your models.
"""

# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)

"""## Fit the model(s)
You're finally ready to fit the models and select the best one!

### Building Training/Test set
"""

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

flights = (spark.read.format("csv")
  .option("inferSchema", "true")
  .option("header", "true")
  .load("./dataset/flights_small.csv"))
flights.createOrReplaceTempView("flights")

planes = (spark.read.format("csv")
  .option("inferSchema", "true")
  .option("header", "true")
  .load('./dataset/planes.csv'))
planes.createOrReplaceTempView("planes")

# Rename year column
planes = planes.withColumnRenamed('year', 'plane_year')

# Join the DataFrame
model_data = flights.join(planes, on='tailnum', how='leftouter')

# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast('integer'))
model_data = model_data.withColumn('air_time', model_data.air_time.cast('integer'))
model_data = model_data.withColumn('month', model_data.month.cast('integer'))
model_data = model_data.withColumn('plane_year', model_data.plane_year.cast('integer'))

# Create the column plane_age
model_data = model_data.withColumn('plane_age', model_data.year - model_data.plane_year)

# Create is_late
model_data = model_data.withColumn('is_late', model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn('label', model_data.is_late.cast('integer'))

# Remove missing values
model_data = model_data.filter('arr_delay is not NULL and dep_delay is not NULL and \
                               air_time is not NULL and plane_year is not NULL')

# Create StringIndexer
carr_indexer = StringIndexer(inputCol='carrier', outputCol='carrier_index')

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol='carrier_index', outputCol='carrier_fact')

# Create a StringIndexer
dest_indexer = StringIndexer(inputCol='dest', outputCol='dest_index')

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol='dest_index', outputCol='dest_fact')

# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=['month', 'air_time', 'carrier_fact', 'dest_fact', 'plane_age'],
                                outputCol='features')

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])

# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)

"""## Evaluate the model
It's finally time to test your model on it! You can use the same evaluator you made to fit the model.
"""

# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))