import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Reading dataset.
main_df = pd.read_csv('Datasets/review_dataset.csv')
df = main_df.copy()


# Defining Wilson Lower Bound method.
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# Inspecting the dataset.
df.head()
df.isnull().sum()
df.dropna(inplace=True)
df['asin'].nunique()
df[['helpful_yes', 'total_vote']].nunique()

# Calculating overall rating' mean.
average_user_raw_overall = df['overall'].mean()

# Latest scores are more important.
time_weighted_overall = df.loc[df['day_diff'] < 31, 'overall'].mean() * 30/100 + \
                        df.loc[(df['day_diff'] < 91) & (df['day_diff'] > 30), 'overall'].mean() * 25/100 + \
                        df.loc[(df['day_diff'] < 181) & (df['day_diff'] > 90), 'overall'].mean() * 20/100 + \
                        df.loc[(df['day_diff'] < 366) & (df['day_diff'] > 180), 'overall'].mean() * 15/100 + \
                        df.loc[df['day_diff'] > 365, 'overall'].mean() * 10/100

# Calculating every weight' mean.
df.loc[df['day_diff'] < 31, 'overall'].mean() # less than 30 days (4.74)
df.loc[(df['day_diff'] < 91) & (df['day_diff'] > 30), 'overall'].mean() # between 30-90 days (4.80)
df.loc[(df['day_diff'] < 181) & (df['day_diff'] > 90), 'overall'].mean() # between 90-180 days (4.65)
df.loc[(df['day_diff'] < 366) & (df['day_diff'] > 180), 'overall'].mean() # between 180-365 days (4.68)
df.loc[df['day_diff'] > 365, 'overall'].mean() # more than 365 days ago  (4.52)

# Creating negative vote column.
df['helpful_no'] = (df['total_vote'] - df['helpful_yes'])

# Inspecting some rating calculation methods. We will use WLB.
df['score_pos_neg_diff'] = (df['helpful_yes'] - df['helpful_no'])
df['score_average_rating'] = df.apply(lambda row: row['helpful_yes'] / row['total_vote'] if row['total_vote'] > 0 else 0, axis=1)
df.describe([0.91, 0.92, 0.93, 0.94, 0.95]).T
df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1)
df.describe([0.91, 0.93, 0.95, 0.97, 0.99]).T

# Sorting results.
df.sort_values('wilson_lower_bound', ascending=False).head(20)
