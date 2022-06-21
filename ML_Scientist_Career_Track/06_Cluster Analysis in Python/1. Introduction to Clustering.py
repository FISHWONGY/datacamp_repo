import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Unsupervised learning: basics
'''
What is unsupervised learning?
        A group of machine learning algorithm that find patterns in data
        Data for algorithms has not been labeled, classified or characterized
        The objective of the algorithm is to interpret any structure in the data
        Common unsupervised learning algorithms : Clustering, neural network, anomaly detection
What is clustering?
        The process of grouping items with similar characteristics
        Items in groups similar to each other than in other groups
        Example: distance between points on a 2D plane
'''

# Pokémon sightings
x = [9, 6, 2, 3, 1, 7, 1, 6, 1, 7, 23, 26, 25, 23, 21, 23, 23, 20, 30, 23]
y = [8, 4, 10, 6, 0, 4, 10, 10, 6, 1, 29, 25, 30, 29, 29, 30, 25, 27, 26, 30]

# Create a scatter plot
plt.scatter(x, y)


# Basics of cluster analysis
'''
What is a cluster?
        A group of items with similar characteristics
        Google News: articles where similar words and word associations appear together
        Customer Segments
Clustering Algorithms
        Hierarchical Clustering
        K-means Clustering
        Other clustering algorithms: DBSCAN, Gaussian Methods
'''

# Pokémon sightings: hierarchical clustering
'''
We are going to continue the investigation into the sightings of legendary Pokémon from the previous exercise. 
Remember that in the scatter plot of the previous exercise, you identified two areas where Pokémon sightings were dense. 
This means that the points seem to separate into two clusters. 
In this exercise, you will form two clusters of the sightings using hierarchical clustering.
'''
df = pd.DataFrame({'x': x, 'y': y})

from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() to compute distance
Z = linkage(df, 'ward')

# Generate cluster labels
df['cluster_labels'] = fcluster(Z, 2, criterion='maxclust')

# Plot the points with seaborn
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)


# Pokémon sightings: k-means clustering
'''
We are going to continue the investigation into the sightings of legendary Pokémon from the previous exercise. 
Just like the previous exercise, we will use the same example of Pokémon sightings. 
In this exercise, you will form clusters of the sightings using k-means clustering.
'''
df = df.astype('float')

from scipy.cluster.vq import kmeans, vq

# Compute cluster centers
centroids, _ = kmeans(df, 2)

# Assign cluster labels
df['cluster_labels'], _ = vq(df, centroids)

# Plot the points with seaborn
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)


# Data preparation for cluster analysis
'''
Why do we need to prepare data for clustering?
    Variables have incomparable units
    Variables with same units have vastly different scales and variances
    Data in raw form may lead to bias in clustering
    Clusters may be heavily dependent on one variable
    Solution: normalization of individual variables
'''

# Normalize basic list data
'''
Now that you are aware of normalization, let us try to normalize some data. 
goals_for is a list of goals scored by a football team in their last ten matches. 
Let us standardize the data using the whiten() function.
'''
from scipy.cluster.vq import whiten

goals_for = [4, 3, 2, 3, 1, 1, 2, 0, 1, 4]

# Use the whiten() function to standardize the data
scaled_data = whiten(goals_for)
print(scaled_data)


# Visualize normalized data
'''
After normalizing your data, you can compare the scaled data to the original data to see the difference.
'''
plt.plot(goals_for, label='original')
plt.plot(scaled_data, label='scaled')
plt.legend()


# Normalization of small numbers
'''
In earlier examples, you have normalization of whole numbers. 
In this exercise, you will look at the treatment of fractional numbers - the change of interest rates 
in the country of Bangalla over the years.
'''


# Prepare data
rate_cuts = [0.0025, 0.001, -0.0005, -0.001, -0.0005, 0.0025, -0.001, -0.0015, -0.001, 0.0005]

# use the whiten() to standardize the data
scaled_data = whiten(rate_cuts)

plt.plot(rate_cuts, label='original')
plt.plot(scaled_data, label='scaled')
plt.legend()


# FIFA 18: Normalize data
'''
FIFA 18 is a football video game that was released in 2017 for PC and consoles. 
The dataset that you are about to work on contains data on the 1000 top individual players in the game. 
You will explore various features of the data as we move ahead in the course. 
In this exercise, you will work with two columns, eur_wage, the wage of a player in Euros and eur_value, 
their current transfer market value.
'''
fifa = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/06_Cluster Analysis in Python/'
                   'data/fifa_18_sample_data.csv')
print(fifa.columns)

# Scale wage and value
fifa['scaled_wage'] = whiten(fifa['eur_wage'])
fifa['scaled_value'] = whiten(fifa['eur_value'])

fifa.plot(x='scaled_wage', y='scaled_value', kind='scatter');

# Check mean and standard deviation of scaled values
print(fifa[['scaled_wage', 'scaled_value']].describe())

