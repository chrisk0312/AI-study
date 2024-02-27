from sklearn.datasets import load_diabetes
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = load_diabetes()
x = np.array(datasets.data)
y = np.array(datasets.target)

import pandas as pd

# Convert to DataFrame
df = pd.DataFrame(data=np.c_[datasets['data'], datasets['target']], columns=datasets['feature_names'] + ['target'])

# 결측치 확인
print(df.isnull())
print(df.isnull().sum())
print(df.info())

#        age    sex    bmi     bp     s1     s2     s3     s4     s5     s6  target
# 0    False  False  False  False  False  False  False  False  False  False   False
# 1    False  False  False  False  False  False  False  False  False  False   False
# 2    False  False  False  False  False  False  False  False  False  False   False
# 3    False  False  False  False  False  False  False  False  False  False   False
# 4    False  False  False  False  False  False  False  False  False  False   False
# ..     ...    ...    ...    ...    ...    ...    ...    ...    ...    ...     ...
# 437  False  False  False  False  False  False  False  False  False  False   False
# 438  False  False  False  False  False  False  False  False  False  False   False
# 439  False  False  False  False  False  False  False  False  False  False   False
# 440  False  False  False  False  False  False  False  False  False  False   False
# 441  False  False  False  False  False  False  False  False  False  False   False

# [442 rows x 11 columns]
# age       0
# sex       0
# bmi       0
# bp        0
# s1        0
# s2        0
# s3        0
# s4        0
# s5        0
# s6        0
# target    0
# dtype: int64
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 442 entries, 0 to 441
# Data columns (total 11 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   age     442 non-null    float64
#  1   sex     442 non-null    float64
#  2   bmi     442 non-null    float64
#  3   bp      442 non-null    float64
#  4   s1      442 non-null    float64
#  5   s2      442 non-null    float64
#  6   s3      442 non-null    float64
#  7   s4      442 non-null    float64
#  8   s5      442 non-null    float64
#  9   s6      442 non-null    float64
#  10  target  442 non-null    float64
# dtypes: float64(11)

print('=====================')
def find_outliers(df):
    # Calculate IQR for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Define a mask for values which are NOT outliers
    mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))

    # Return rows with outliers
    return df[~mask].dropna(how='all')

# Use the function to find outliers
outliers = find_outliers(df)
print(outliers)

#      age  sex       bmi  bp        s1        s2        s3        s4        s5        s6  target
# 23   NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN  0.133597  0.135612     NaN
# 35   NaN  NaN       NaN NaN       NaN       NaN  0.133318       NaN       NaN       NaN     NaN
# 58   NaN  NaN       NaN NaN       NaN       NaN  0.181179       NaN       NaN       NaN     NaN
# 84   NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN       NaN -0.129483     NaN
# 117  NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN       NaN  0.135612     NaN
# 123  NaN  NaN       NaN NaN  0.152538  0.198788       NaN  0.185234       NaN       NaN     NaN
# 141  NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN       NaN  0.131470     NaN
# 161  NaN  NaN       NaN NaN  0.133274  0.131461       NaN       NaN       NaN       NaN     NaN
# 168  NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN       NaN  0.127328     NaN
# 169  NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN  0.133597       NaN     NaN
# 202  NaN  NaN       NaN NaN  0.126395       NaN       NaN       NaN       NaN       NaN     NaN
# 230  NaN  NaN       NaN NaN  0.153914  0.155887       NaN       NaN       NaN       NaN     NaN
# 245  NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN       NaN -0.129483     NaN
# 248  NaN  NaN       NaN NaN  0.127771  0.128016       NaN       NaN       NaN       NaN     NaN
# 256  NaN  NaN  0.160855 NaN       NaN       NaN       NaN       NaN       NaN       NaN     NaN
# 260  NaN  NaN       NaN NaN       NaN       NaN  0.151726       NaN       NaN       NaN     NaN
# 261  NaN  NaN       NaN NaN       NaN       NaN  0.177497       NaN       NaN       NaN     NaN
# 269  NaN  NaN       NaN NaN       NaN       NaN  0.159089       NaN       NaN       NaN     NaN
# 276  NaN  NaN       NaN NaN  0.125019       NaN       NaN       NaN       NaN       NaN     NaN
# 286  NaN  NaN       NaN NaN       NaN       NaN  0.140681       NaN       NaN       NaN     NaN
# 287  NaN  NaN       NaN NaN  0.125019  0.125198       NaN       NaN       NaN       NaN     NaN
# 322  NaN  NaN       NaN NaN       NaN       NaN       NaN  0.155345  0.133397       NaN     NaN
# 346  NaN  NaN       NaN NaN  0.127771  0.127390       NaN       NaN       NaN       NaN     NaN
# 350  NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN       NaN  0.135612     NaN
# 353  NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN  0.132376       NaN     NaN
# 366  NaN  NaN  0.137143 NaN       NaN       NaN       NaN       NaN       NaN       NaN     NaN
# 367  NaN  NaN  0.170555 NaN       NaN       NaN       NaN       NaN       NaN       NaN     NaN
# 376  NaN  NaN       NaN NaN       NaN  0.130208       NaN       NaN       NaN       NaN     NaN
# 406  NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN       NaN -0.137767     NaN
# 428  NaN  NaN       NaN NaN       NaN       NaN       NaN       NaN       NaN  0.131470     NaN
# 441  NaN  NaN       NaN NaN       NaN       NaN  0.173816       NaN       NaN       NaN     NaN


import matplotlib.pyplot as plt
plt.boxplot(df)
plt.show()