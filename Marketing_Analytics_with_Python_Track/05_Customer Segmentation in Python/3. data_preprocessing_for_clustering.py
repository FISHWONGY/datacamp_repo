import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

datamart = pd.read_csv('./datasets/chapter_3/rfm_datamart.csv')
datamart.head()

sns.distplot(datamart['Recency'])
plt.show()

sns.distplot(datamart['Frequency'])
plt.show()

sns.distplot(datamart['MonetaryValue'])
plt.show()

import numpy as np
frequency_log = np.log(datamart['Frequency'])

sns.distplot(frequency_log)
plt.show()

"""## Centering and Scaling Variables

**Identifying an issue**
- Analyzing key statistics of the dataset
- Compare mean and standard deviation

"""

datamart.describe()

"""**Centering variables**

Centering variables is done by substracting the average value from each observation
"""

datamart_centered = datamart - datamart.mean()
datamart_centered.describe().round(2)

"""**Scaling variables with different variance**
- K-means works better on variables with the same variance / standard deviation
- Scaling variables is done by dividing them by standard deviation of each

"""

datamart_scaled = datamart / datamart.std()
datamart_scaled.describe().round(2)

datamart_normalized = (datamart - datamart.mean()) / datamart.std()
datamart_normalized.describe().round(2)

"""**Combining centering and scaling**
- Substract mean and divide by std manually (above examples)
- Use *scaler* from *scikit-learn* library
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(datamart)

datamart_normalized = scaler.transform(datamart)

print('mean: ', datamart_normalized.mean(axis = 0).round(2))
print('std: ', datamart_normalized.std(axis = 0).round(2))

data_normalized = pd.DataFrame(datamart_normalized, index = datamart.index, columns = datamart.columns)
data_normalized.describe().round(2)

"""**Pre-processing pipeline**
1. Unskew the data - log transformation
2. Standarize to the same average values
3. Scale to the same standard deviation
4. Store as a separate array to be used for clustering

**Coding Sequence*

1. Unskew the data with log transformation
```python
import numpy as np
datamart_log = np.log(datamart)
```

2. Normalize the variable with *StandardScaler*
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(datamart)
datamart_normalized = scaler.transform(datamart_log)
```
"""