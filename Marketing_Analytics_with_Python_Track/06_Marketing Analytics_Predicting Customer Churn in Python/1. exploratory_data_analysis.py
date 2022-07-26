import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

telco = pd.read_csv('./datasets/Churn.csv')
print(telco.head())

"""**Here's the dataset information**"""

telco.info()

"""**There is a Churn feature in the dataset, let's see whic values it includes**"""

telco['Churn'].value_counts()

"""**Group by Churn and calculate the mean**"""

print(telco.groupby(['Churn']).mean())

"""**Let's group by state to see the churn values**"""

print(telco.groupby('State')['Churn'].value_counts())

"""**Here is a distribution plot of the Account Length feature**"""

sns.distplot(telco['Account_Length'])
plt.show()

sns.boxplot(x = 'Churn',
            y = 'Account_Length',
            data = telco,
            sym = "")

plt.show()

sns.boxplot(x = 'Churn',
            y = 'Account_Length',
            data = telco,
            hue = 'Intl_Plan')

plt.show()

"""**Do customers who have international plans make more customer service calls? Or do they tend to churn more? How about voicemail plans? Let's find out!**"""

sns.boxplot(x = 'Churn',
            y = 'CustServ_Calls',
            data = telco,
            sym = '',
            hue = 'Intl_Plan')

plt.show()

"""**Ok, we see that the customers that churn make more customer service calls. Unless they have an international plan, in which case they leave fewer customer service calls**"""