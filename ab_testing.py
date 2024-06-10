import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind

pd.set_option('float_format', lambda x: '%.5f' % x)

# Reading datasets
main_df_control = pd.read_excel('Datasets/ab_dataset.xlsx', sheet_name='Control Group')
df_c = main_df_control.copy()
main_df_test = pd.read_excel('Datasets/ab_dataset.xlsx', sheet_name='Test Group')
df_t = main_df_test.copy()

# Inspecting datasets
df_c.head()
df_c.shape
df_c.describe().T

df_t.head()
df_t.shape
df_t.describe().T

# AB test hypothesis
# Hypothesis: Is there a statistically significant difference between the old and new systems?

# Testing for normality.
test_stat, pvalue = shapiro(df_c.loc[:, 'Purchase'])
print('Test stat = %.4f, p-value= %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df_t.loc[:, 'Purchase'])
print('Test stat = %.4f, p-value= %.4f' % (test_stat, pvalue))
# Both normally distributed.

# Testing for equal variances.
test_stat, pvalue = levene(df_c.loc[:, 'Purchase'],
                           df_t.loc[:, 'Purchase'])
print('Test stat = %.4f, p-value= %.4f' % (test_stat, pvalue))
# Equal variances.

# T-test
test_stat, pvalue = ttest_ind(df_c.loc[:, 'Purchase'],
                              df_t.loc[:, 'Purchase'],
                              equal_var=True)
print('Test stat = %.4f, p-value= %.4f' % (test_stat, pvalue))

# We don't reject null hypothesis
