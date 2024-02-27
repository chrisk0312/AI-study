from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import time
import math
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR

#data
path = "C:\\_data\\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual','registered','count'],axis=1)
y = train_csv['count']

print(x.shape, y.shape) #(10886, 8) (10886,)

# 결측치 확인
print(train_csv.isnull())
print(train_csv.isnull().sum())
print(train_csv.info())

#                      season  holiday  workingday  weather   temp  atemp  humidity  windspeed  casual  registered  count
# datetime
# 2011-01-01 00:00:00   False    False       False    False  False  False     False      False   False       False  False
# 2011-01-01 01:00:00   False    False       False    False  False  False     False      False   False       False  False
# 2011-01-01 02:00:00   False    False       False    False  False  False     False      False   False       False  False
# 2011-01-01 03:00:00   False    False       False    False  False  False     False      False   False       False  False
# 2011-01-01 04:00:00   False    False       False    False  False  False     False      False   False       False  False
# ...                     ...      ...         ...      ...    ...    ...       ...        ...     ...         ...    ...
# 2012-12-19 19:00:00   False    False       False    False  False  False     False      False   False       False  False
# 2012-12-19 20:00:00   False    False       False    False  False  False     False      False   False       False  False
# 2012-12-19 21:00:00   False    False       False    False  False  False     False      False   False       False  False
# 2012-12-19 22:00:00   False    False       False    False  False  False     False      False   False       False  False
# 2012-12-19 23:00:00   False    False       False    False  False  False     False      False   False       False  False

# [10886 rows x 11 columns]
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# casual        0
# registered    0
# count         0
# dtype: int64
# <class 'pandas.core.frame.DataFrame'>
# Index: 10886 entries, 2011-01-01 00:00:00 to 2012-12-19 23:00:00
# Data columns (total 11 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   season      10886 non-null  int64
#  1   holiday     10886 non-null  int64
#  2   workingday  10886 non-null  int64
#  3   weather     10886 non-null  int64
#  4   temp        10886 non-null  float64
#  5   atemp       10886 non-null  float64
#  6   humidity    10886 non-null  int64
#  7   windspeed   10886 non-null  float64
#  8   casual      10886 non-null  int64
#  9   registered  10886 non-null  int64
#  10  count       10886 non-null  int64
# dtypes: float64(3), int64(8)

print('=====================')
def find_outliers(train_csv):
    # Calculate IQR for each column
    Q1 = train_csv.quantile(0.25)
    Q3 = train_csv.quantile(0.75)
    IQR = Q3 - Q1

    # Define a mask for values which are NOT outliers
    mask = ~((train_csv < (Q1 - 1.5 * IQR)) | (train_csv > (Q3 + 1.5 * IQR)))

    # Return rows with outliers
    return train_csv[~mask].dropna(how='all')

# Use the function to find outliers
outliers = find_outliers(x)
print(outliers)

# season  holiday  workingday  weather  temp  atemp  humidity  windspeed
# datetime
# 2011-01-08 14:00:00     NaN      NaN         NaN      NaN   NaN    NaN       NaN    32.9975
# 2011-01-08 17:00:00     NaN      NaN         NaN      NaN   NaN    NaN       NaN    36.9974
# 2011-01-09 09:00:00     NaN      NaN         NaN      NaN   NaN    NaN       NaN    35.0008
# 2011-01-09 11:00:00     NaN      NaN         NaN      NaN   NaN    NaN       NaN    35.0008
# 2011-01-12 12:00:00     NaN      NaN         NaN      NaN   NaN    NaN       NaN    39.0007
# ...                     ...      ...         ...      ...   ...    ...       ...        ...
# 2012-11-12 22:00:00     NaN      1.0         NaN      NaN   NaN    NaN       NaN        NaN
# 2012-11-12 23:00:00     NaN      1.0         NaN      NaN   NaN    NaN       NaN        NaN
# 2012-11-13 01:00:00     NaN      NaN         NaN      NaN   NaN    NaN       NaN    43.0006
# 2012-12-05 14:00:00     NaN      NaN         NaN      NaN   NaN    NaN       NaN    32.9975
# 2012-12-18 15:00:00     NaN      NaN         NaN      NaN   NaN    NaN       NaN    32.9975

import matplotlib.pyplot as plt
plt.boxplot(train_csv)
plt.show()