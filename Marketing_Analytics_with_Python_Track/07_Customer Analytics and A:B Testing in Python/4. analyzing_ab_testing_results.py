import pandas as pd
import stats
ab_test_results = pd.read_csv('some.csv')
# Find the unique users in each group, by device and gender
results = ab_test_results.groupby(by=['group', 'device', 'gender']).agg({'uid': pd.Series.nunique}) 

# Find the overall number of unique users using "len" and "unique"
unique_users = len(ab_test_results.uid.unique())

# Find the percentage in each group
results = results / unique_users * 100
print(results)

"""## Understanting Statistical Significance


"""


def get_pvalue(con_conv, test_conv, con_size, test_size):  
    lift = - abs(test_conv - con_conv)

    scale_one = con_conv * (1 - con_conv) * (1 / con_size)
    scale_two = test_conv * (1 - test_conv) * (1 / test_size)
    scale_val = (scale_one + scale_two)**0.5

    p_value = 2 * stats.norm.cdf(lift, loc = 0, scale = scale_val )

    return p_value

# Get the p-value
p_value = get_pvalue(con_conv=0.48, test_conv=0.5, con_size=1_000, test_size=1_000)
print(p_value)

# Compute the p-value
p_value = get_pvalue(con_conv=cont_conv, test_conv=test_conv, con_size=cont_size, test_size=test_size)
print(p_value)

# Check for statistical significance
if p_value >= 0.05:
    print("Not Significant")
else:
    print("Significant Result")


## Confidence Interval
def get_ci(value, cl, sd):
  loc = sci.norm.ppf(1 - cl/2)
  rng_val = sci.norm.cdf(loc - value/sd)

  lwr_bnd = value - rng_val
  upr_bnd = value + rng_val 

  return_val = (lwr_bnd, upr_bnd)
  return(return_val)


# Compute and print the confidence interval
confidence_interval = get_ci(1, 0.95, 2)
print(confidence_interval)

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

test_var = 1.6255542142857143e-06
cont_var = 1.411507925080655e-06
test_conv = 0.102005
cont_conv = 0.090965

# Compute the standard deviations
control_sd = cont_var**0.5
test_sd = test_var**0.5

# Create the range of x values 
control_line = np.linspace(cont_conv - 3 * control_sd, cont_conv + 3 * control_sd , 100)
test_line = np.linspace(test_conv - 3 * test_sd,  test_conv + 3 * test_sd , 100)

# Plot the distribution 
plt.plot(control_line, matplotlib.mlab.normpdf(control_line, cont_conv, control_sd))
plt.plot(test_line, matplotlib.mlab.normpdf(test_line,test_conv, test_sd))
plt.show()

# Find the lift mean and standard deviation
lift_mean = test_conv - cont_conv
lift_sd = (test_var + cont_var) ** 0.5

# Generate the range of x-values
lift_line = np.linspace(lift_mean - 3 * lift_sd, lift_mean + 3 * lift_sd, 100)

# Plot the lift distribution
plt.plot(lift_line, mlab.normpdf(lift_line, lift_mean, lift_sd))

# Add the annotation lines
plt.axvline(x=lift_mean, color='green')
plt.axvline(x=lwr_ci, color='red')
plt.axvline(x=upr_ci, color='red')
plt.show()
