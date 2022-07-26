import pandas as pd
from sklearn.cluster import KMeans

datamart_normalized = pd.read_csv('./datasets/chapter_4/datamart_normalized_df.csv')
datamart_rfm = pd.read_csv('./datasets/chapter_4/datamart_rfm.csv')

datamart_normalized.head()

kmeans = KMeans(n_clusters = 3, random_state = 1)
kmeans.fit(datamart_normalized)
cluster_labels = kmeans.labels_

datamart_rfm_k3 = datamart_rfm.assign(Cluster = cluster_labels)
grouped = datamart_rfm_k3.groupby(['Cluster'])
grouped.agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
}).round(1)

"""## Choosing number of clusters

**Elbow Criterion Method**
- Plot the number of clusters against within-clusters sum-of-squared-errors (SSE)
    - sum of squared distances from every data point to their cluster center
- Identify the "elbow" in the plot
    - where the decrease in SSE slows down and becomes somewhat marginal
    - shows where there are diminishing returns by increasing the number of clusters
- "Elbow" - a point representing an "optimal" number of clusters from a sum-of-squared errors perspective

```python
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import pyplot as plt

#fit KMeans and calculate SSE for each *k*
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, random_state = 1)
    kmeans.fit(data_normalized)
    sse[k] = kmeans.inertia_ #sum of squared distances to closest cluster center

#Plot SSE for each *k*
plt.title('The Elbow Method")
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x = list(sse.keys()), y = list(sse.values()))
plt.show()
```
**Analyze segments**
- Build clustering at and around elbow solution
- Analyze their properties - average RFM values
- Compare against each other and choose one which makes most business sense

**Approaches to build customer personas**
- Summary statistics for each cluster e.g. average RFM values
- Snake plots
- Relative importance of cluster attributes compared to population

**Snake Plots**
- Used to compare different segments

transform *datamart_normalized* as DataFrame and add a *Cluster* column

```python
datamart_normalized = pd.DataFrame(datamart_normalized,
                                   index = datamart_rfm.index,
                                   columns = datamart_rfm.columns)
datamart_normalized['Cluster'] = datamart_rfm_k3['Cluster']
```

Melt data into a long format so RFM values ansd metric names are stored in 1 column each

```python
datamart_melt = pd.melt(datamart_normalized.reset_index(),
                        id_vars = ['CustomerID', 'Cluster'],
                        value_vars = ['Recency', 'Frequency', 'MonetaryValue'],
                        var_name = 'Attribute',
                        value_name = 'Value')
```

Visualize the Snake Plot

```python
plt.title('Snake plot of standarized variables')
sns.lineplot(x = "Attribute", y = "Value", hue = 'Cluster', data = datamart_melt)
plt.show()
```

**Analyze and plot relative importance**
- The further a ratio s from 0, the more important that attribute is for a segment relative o the total population

```python
cluster_avg = datamart_rfm_k3.groupby(['Cluster']).mean()
population_avg = datamart_rfm.mean()
relative_imp = cluster_avg / population_avg - 1
relative_imp.round(2)

#heatmap
plt.figure(figsize=(8,2))
plt.title('Relative importance of attributes')
sns.heatmap(data = relative_imp, annot = True, fmt = '.2f', cmap = 'RdYlGn')
plt.show()
```
"""

datamart_normalized['Cluster'] = datamart_rfm_k3['Cluster']
datamart_normalized.head()

datamart_melt = pd.melt(datamart_normalized.reset_index(),
                        id_vars = ['CustomerID', 'Cluster'],
                        value_vars = ['Recency', 'Frequency', 'MonetaryValue'],
                        var_name = 'Metric',
                        value_name = 'Value')
datamart_melt.head()

import seaborn as sns
from matplotlib import pyplot as plt

plt.title('Snake plot of normalized variables')
plt.xlabel('Metric')
plt.ylabel('Value')
sns.lineplot(x = "Metric", y = "Value", hue = 'Cluster', data = datamart_melt)
plt.show()

cluster_avg = datamart_rfm_k3.groupby(['Cluster']).mean()
population_avg = datamart_rfm.mean()
relative_imp = cluster_avg / population_avg - 1
print(relative_imp.round(2))

plt.figure(figsize = (8,2))
plt.title('Relative importance of attributes')
sns.heatmap(data = relative_imp, annot = True, fmt = '.2f', cmap = 'RdYlGn')
plt.show()