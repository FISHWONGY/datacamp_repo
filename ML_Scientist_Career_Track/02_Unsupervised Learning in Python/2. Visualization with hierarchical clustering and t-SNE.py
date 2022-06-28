import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing hierarchies
'''
Visualizations communicate insight
        't-SNE': Creates a 2D map of a dataset
        'Hierarchical clustering'
A hierarchy of groups
        Groups of living things can form a hierarchy
        Cluster are contained in one another
Hierarchical clustering
        Every element begins in a separate cluster
        At each step, the two closest clusters are merged
        Continue until all elements in a single cluster
        This is "agglomerative"(or divisive) hierarchical clustering
'''

# Hierarchical clustering of the grain data
'''
In the video, you learned that the SciPy linkage() function performs hierarchical clustering on an array of samples. 
Use the linkage() function to obtain a hierarchical clustering of the grain samples, and use dendrogram() to visualize the result. 
A sample of the grain measurements is provided in the array samples, while the variety of each grain sample is given by the list varieties.
'''
df = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                 '02_Unsupervised Learning in Python/data/seeds.csv', header=None)
df[7] = df[7].map({1: 'Kama wheat', 2: 'Rosa wheat', 3: 'Canadian wheat'})
print(df.head())
df = df.iloc[0: 209]


samples = df.iloc[:, :-1].values
varieties = df.iloc[:, -1].values

from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
plt.figure(figsize=(15, 5))
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
          );


# Hierarchies of stocks
'''
In chapter 1, you used k-means clustering to cluster companies according to their stock price movements. 
Now, you'll perform hierarchical clustering of the companies. 
You are given a NumPy array of price movements movements, where the rows correspond to companies, 
and a list of the company names companies. SciPy hierarchical clustering doesn't fit into a sklearn pipeline, 
so you'll need to use the normalize() function from sklearn.preprocessing instead of Normalizer.
'''
df = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                 '02_Unsupervised Learning in Python/data/company-stock-movements-2010-2015-incl.csv', index_col=0)
print(df.head())


movements = df.values
companies = df.index.values

from sklearn.preprocessing import normalize

# Normalize the movements: normalize_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
plt.figure(figsize=(15, 5))
dendrogram(mergings,
           labels=companies,
           leaf_rotation=90,
           leaf_font_size=6);


# Cluster labels in hierarchical clustering
'''
Intermediate clusterings & height on dendrogram
        Height on dendrogram specifies max. distance between merging clusters
        Don't merge clusters further apart than this.
Distance between clusters
        Defined by "linkage method"
        In "complete" linkage: distance between clusters is max. distance between their samples
        Different linkage method, different hierarchical clustering
'''

# Different linkage, different hierarchical clustering!
'''
In the video, you saw a hierarchical clustering of the voting countries at the Eurovision song contest using 'complete' linkage. 
Now, perform a hierarchical clustering of the voting countries with 'single' linkage, and compare the resulting dendrogram with the one in the video. Different linkage, different hierarchical clustering!

You are given an array samples. Each row corresponds to a voting country, 
and each column corresponds to a performance that was voted for. 
The list country_names gives the name of each voting country. This dataset was obtained from Eurovision.
'''

df = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                 '02_Unsupervised Learning in Python/data/eurovision-2016.csv')
print(df)

samples = df.iloc[:, 2:7].values[:42]
country_names = df.iloc[:, 1].values[:42]

# Calculate the linkage: mergings
mergings = linkage(samples, method='single')

# Plot the dendrogram
plt.figure(figsize=(15, 5))
dendrogram(mergings,
           labels=country_names,
           leaf_rotation=90,
           leaf_font_size=6);


# Extracting the cluster labels
'''
In the previous exercise, you saw that the intermediate clustering of the grain samples at height 6 has 3 clusters. 
Now, use the fcluster() function to extract the cluster labels for this intermediate clustering, 
and compare the labels with the grain varieties using a cross-tabulation.
'''
df = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                 '02_Unsupervised Learning in Python/data/seeds.csv', header=None)
df[7] = df[7].map({1: 'Kama wheat', 2: 'Rosa wheat', 3: 'Canadian wheat'})
print(df.head())
df = df.iloc[0:209]


samples = df.iloc[:, :-1].values
varieties = df.iloc[:, -1].values

from scipy.cluster.hierarchy import fcluster

mergings = linkage(samples, method='complete')

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)


# t-SNE for 2-dimensional maps
'''
t-SNE for 2-dimensional maps
    t-SNE = "t-distributed stochastic neighbor embedding"
    Maps samples to 2D space (or 3D)
    Map approximately preserves nearness of samples
    Great for inspecting dataset
'''

# t-SNE visualization of grain dataset
df = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                 '02_Unsupervised Learning in Python/data/seeds.csv', header=None)
df = df.iloc[0:209]

samples = df.iloc[:, :-1].values
variety_numbers = df.iloc[:, -1].values

from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:, 0]

# Select the 1st feature: ys
ys = tsne_features[:, 1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers);


# A t-SNE map of the stock market
'''
t-SNE provides great visualizations when the individual samples can be labeled. 
In this exercise, you'll apply t-SNE to the company stock price data. 
A scatter plot of the resulting t-SNE features, labeled by the company names, gives you a map of the stock market! 
The stock price movements for each company are available as the array normalized_movements (these have already been normalized for you). 
The list companies gives the name of each company.
'''
df = pd.read_csv('./datacamp_repo/ML_Scientist_Career_Track/'
                 '02_Unsupervised Learning in Python/data/company-stock-movements-2010-2015-incl.csv', index_col=0)
movements = df.values
companies = df.index.values
normalized_movements = normalize(movements)

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:, 0]

# Select the 1st feature: ys
ys = tsne_features[:, 1]

# Scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=8, alpha=0.75)

