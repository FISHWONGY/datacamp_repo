import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

marketing = pd.read_csv('./datasets/marketing.csv')

# let's first isolate the rows of the DS where the channel is Email
email = marketing[marketing['marketing_channel'] == 'Email']

alloc = email.groupby(['variant'])['user_id'].nunique()

alloc.plot(kind='bar')
plt.title('Personalization test allocation')
plt.xticks(rotation = 0)
plt.ylabel('# participants')
plt.show()

subscribers = email.groupby(['user_id', 'variant'])['converted'].max()

subscribers_df = pd.DataFrame(subscribers.unstack(level = 1))

control = subscribers_df['control'].dropna()

personalization = subscribers_df['personalization'].dropna()

print(subscribers_df)

print('Control conversion rate:', np.mean(control))
print('Personalization conversion rate:', np.mean(personalization))

"""## Calculating Lift and Statistical Significance

**Lift is the difference between the A and B conversion rates divided by B conversion rate. In this case it would be:**

\begin{equation*}
\text{Lift} = \frac{\text{Treatment conversion rate - Control conversion rate}}{\text{Control conversion rate}}
\end{equation*}

**The result is the relative percent difference of treatment compared to control**
"""

# Calculate the mean or conversion rate of control and personalization groups
a_mean = np.mean(control)
b_mean = np.mean(personalization)

# Calculate the lift
lift = (b_mean - a_mean) / a_mean

print("lift:", str(round(lift*100,2)) + '%')

"""**Statistical significance is vital to understand if a test showed positive result by chance or 
if it's reflective of a true difference between variants**"""

from scipy.stats import ttest_ind

t = ttest_ind(control, personalization)

print(t)

"""**Given that p is 0.006, we can conclude that the results are statistically significant**

### Building an A/B test segmentation function
"""


def ab_segmentation(segment):
    # loop for each subsegment in marketing
    for subsegment in np.unique(marketing[segment].values):
        
        print(subsegment)
        
        email = marketing[(marketing['marketing_channel'] == 'Email') & (marketing[segment] == subsegment)]
        
        subscribers = email.groupby(['user_id', 'variant'])['converted'].max()
        subscribers = pd.DataFrame(subscribers.unstack())
        
        control = subscribers['control'].dropna()
        personalization = subscribers['personalization'].dropna()
        
        a_mean = np.mean(control)
        b_mean = np.mean(personalization)
        
        lift = (b_mean - a_mean)/ a_mean
        
        print('lift:', lift)
        print('t-statistic:', ttest_ind(control, personalization), '\n\n')


ab_segmentation('language_displayed')

ab_segmentation('age_group')
