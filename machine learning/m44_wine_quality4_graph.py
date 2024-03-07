from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler

path = "C:\\_data\\DACON\\와인품질분류\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]

train_csv.loc[train_csv['type'] == 'red', 'type'] = 1
train_csv.loc[train_csv['type'] == 'white', 'type'] = 0
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

# x = train_csv.drop(['quality'],axis=1)
# y = train_csv['quality']

# print(np.unique(y,return_counts=True))
# # (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
# from sklearn.preprocessing import LabelEncoder
# import warnings
# warnings.filterwarnings('ignore')

# y = LabelEncoder().fit_transform(y)
# print(np.unique(y,return_counts=True))

plt_df = train_csv.groupby('quality').count()
print(plt_df.head(10))
print(np.arange(len(plt_df.index)))

plt.bar(plt_df.index,plt_df['type'])
plt.xticks(plt_df.index)
# plt.bar(train_csv)
plt.show()

