#https://dacon.io/competitions/official/236214/data

from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from keras.callbacks import EarlyStopping
import time
import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

#대출기간 처리
train_loan_time = train_csv['대출기간']
train_loan_time = train_loan_time.str.split()
for i in range(len(train_loan_time)):
    train_loan_time.iloc[i] = int((train_loan_time)[i][0])
    
#print(train_loan_time)   

test_loan_time = test_csv['대출기간']
test_loan_time = test_loan_time.str.split()
for i in range(len(test_loan_time)):
    test_loan_time.iloc[i] = int((test_loan_time)[i][0])

train_csv['대출기간'] = train_loan_time
test_csv['대출기간'] = test_loan_time

le = LabelEncoder()

#print(test_csv)

le.fit(train_csv['대출기간'])
train_csv['대출기간'] = le.transform(train_csv['대출기간'])
le.fit(test_csv['대출기간'])
test_csv['대출기간'] = le.transform(test_csv['대출기간'])

#근로기간 처리
train_working_time = train_csv['근로기간']
test_working_time = test_csv['근로기간']

for i in range(len(train_working_time)):
    data = train_working_time.iloc[i]
    if data == 'Unknown':
        train_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        train_working_time.iloc[i] = int(30)
    elif data == '< 1 year' or data == '<1 year':
        train_working_time.iloc[i] = int(0)
    else:
        train_working_time.iloc[i] = int(data.split()[0])
    
train_working_time = train_working_time.fillna(train_working_time.mean())

for i in range(len(test_working_time)):
    data = test_working_time.iloc[i]
    if data == 'Unknown':
        test_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        test_working_time.iloc[i] = int(30)
    elif data == '< 1 year' or data == '<1 year':
        test_working_time.iloc[i] = int(0)
    else:
        test_working_time.iloc[i] = int(data.split()[0])
    
test_working_time = test_working_time.fillna(test_working_time.mean())

train_csv['근로기간'] = train_working_time
test_csv['근로기간'] = test_working_time 
#print(test_csv['근로기간'])

le.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le.transform(train_csv['주택소유상태'])
le.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] = le.transform(test_csv['주택소유상태'])

le.fit(train_csv['대출목적'])
train_csv['대출목적'] = le.transform(train_csv['대출목적'])
le.fit(test_csv['대출목적'])
test_csv['대출목적'] = le.transform(test_csv['대출목적'])

x = train_csv.drop(['대출등급'], axis = 1)
mms = MinMaxScaler()
mms.fit(x)
x = mms.transform(x)
test_csv=mms.transform(test_csv)

#print(x)
y = train_csv['대출등급']
y = le.fit_transform(y)

y= y.reshape(-1,1)
ohe = OneHotEncoder(sparse= False)
y = ohe.fit_transform(y)

print(np.unique(y, return_counts= True)) #Name: 근로기간, Length: 96294, dtype: float64

# Check for missing values in train_csv
print(train_csv.isnull().sum())
# (array([0., 1.]), array([577764,  96294], dtype=int64))
# 대출금액            0
# 대출기간            0
# 근로기간            0
# 주택소유상태          0
# 연간소득            0
# 부채_대비_소득_비율     0
# 총계좌수            0
# 대출목적            0
# 최근_2년간_연체_횟수    0
# 총상환원금           0
# 총상환이자           0
# 총연체금액           0
# 연체계좌수           0
# 대출등급            0
# dtype: int64
print('=====================')

def find_outliers(train_csv):
    # Select only numeric columns
    numeric_cols = train_csv.select_dtypes(include=[np.number])

    # Calculate IQR for each column
    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    IQR = Q3 - Q1

    # Define a mask for values which are NOT outliers
    mask = ~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR)))

    # Return rows with outliers
    return numeric_cols[~mask].dropna(how='all')

# Use the function to find outliers
outliers = find_outliers(train_csv)
print(outliers)
#              대출금액  대출기간  근로기간  주택소유상태         연간소득  부채_대비_소득_비율  총계좌수  대출목적  최근_2년간_연체_횟수  총상환원
# 금      총상환이자  총연체금액  연체계좌수
# ID
# TRAIN_00001   NaN   NaN   NaN     NaN          NaN          NaN   NaN  10.0           NaN    NaN        NaN    NaN    NaN
# TRAIN_00004   NaN   NaN   NaN     NaN          NaN          NaN   NaN   8.0           NaN    NaN        NaN    NaN    NaN
# TRAIN_00005   NaN   NaN   NaN     NaN          NaN          NaN   NaN  11.0           NaN    NaN        NaN    NaN    NaN
# TRAIN_00006   NaN   NaN   NaN     NaN          NaN          NaN   NaN  11.0           NaN    NaN        NaN    NaN    NaN
# TRAIN_00010   NaN   NaN   NaN     NaN          NaN          NaN   NaN   NaN           NaN    NaN  1523172.0    NaN    NaN
# ...           ...   ...   ...     ...          ...          ...   ...   ...           ...    ...        ...    ...    ...
# TRAIN_96286   NaN   NaN   NaN     NaN  244800000.0          NaN   NaN   NaN           NaN    NaN  2075832.0    NaN    NaN
# TRAIN_96288   NaN   NaN   NaN     NaN          NaN          NaN   NaN   NaN          10.0    NaN        NaN    NaN    NaN
# TRAIN_96289   NaN   NaN   NaN     NaN  210000000.0          NaN   NaN   NaN           NaN    NaN        NaN    NaN    NaN
# TRAIN_96290   NaN   NaN   NaN     NaN          NaN          NaN   NaN  10.0           NaN    NaN        NaN    NaN    NaN
# TRAIN_96292   NaN   NaN   NaN     NaN          NaN          NaN   NaN   NaN           2.0    NaN        NaN    NaN    NaN

import matplotlib.pyplot as plt

# Select only numeric columns
numeric_cols = train_csv.select_dtypes(include=[np.number])

# Create a boxplot
plt.boxplot(numeric_cols)
plt.show()
