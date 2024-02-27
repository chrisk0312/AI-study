from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR

#data
path = "C:\\_data\\dacon\\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  
test_csv = pd.read_csv(path+"test.csv",index_col=0)         
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'],axis=1) #count 를 드랍, axis=0은 행, axis=1은 열
y = train_csv['count']

# 결측치 확인
print(train_csv.isnull())
print(train_csv.isnull().sum())
print(train_csv.info())

#        hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  ...  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5  count
# id                                                                             ...
# 3     False                 False                   False               False  ...           False          False           False  False  
# 6     False                 False                   False               False  ...           False          False           False  False  
# 7     False                 False                   False               False  ...           False          False           False  False  
# 8     False                 False                   False               False  ...           False          False           False  False  
# 9     False                 False                   False               False  ...           False          False           False  False  
# ...     ...                   ...                     ...                 ...  ...             ...            ...             ...    ...  
# 2174  False                 False                   False               False  ...           False          False           False  False  
# 2175  False                 False                   False               False  ...           False          False           False  False  
# 2176  False                 False                   False               False  ...           False          False           False  False  
# 2178  False                 False                   False               False  ...           False          False           False  False  
# 2179  False                 False                   False               False  ...           False          False           False  False  

# [1459 rows x 10 columns]
# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# count                     0
# dtype: int64
# <class 'pandas.core.frame.DataFrame'>
# Index: 1459 entries, 3 to 2179
# Data columns (total 10 columns):
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1459 non-null   float64
#  2   hour_bef_precipitation  1459 non-null   float64
#  3   hour_bef_windspeed      1459 non-null   float64
#  4   hour_bef_humidity       1459 non-null   float64
#  5   hour_bef_visibility     1459 non-null   float64
#  6   hour_bef_ozone          1459 non-null   float64
#  7   hour_bef_pm10           1459 non-null   float64
#  8   hour_bef_pm2.5          1459 non-null   float64
#  9   count                   1459 non-null   float64
# dtypes: float64(9), int64(1)


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

#       hour  hour_bef_temperature  hour_bef_precipitation  ...  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5
# id                                                        ...
# 3      NaN                   NaN                     1.0  ...             NaN            NaN             NaN
# 8      NaN                   NaN                     NaN  ...             NaN            NaN            64.0
# 27     NaN                   NaN                     NaN  ...             NaN            NaN            84.0
# 29     NaN                   NaN                     1.0  ...             NaN            NaN             NaN
# 32     NaN                   NaN                     1.0  ...             NaN            NaN             NaN
# ...    ...                   ...                     ...  ...             ...            ...             ...
# 2130   NaN                   NaN                     NaN  ...             NaN          118.0            74.0
# 2138   NaN                   NaN                     NaN  ...           0.125            NaN             NaN
# 2142   NaN                   NaN                     NaN  ...           0.112            NaN             NaN
# 2161   NaN                   NaN                     1.0  ...             NaN            NaN             NaN
# 2171   NaN                   NaN                     NaN  ...             NaN            NaN            69.0

# import matplotlib.pyplot as plt
# plt.boxplot(train_csv)
# plt.show()


x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,shuffle=False,random_state=333)

#model
model = Sequential()
model.add(Dense(512,input_dim=9,activation='sigmoid'))
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))


#compile & fit
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=1024,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=102,batch_size=16,validation_split=0.35,verbose=2,callbacks=[es])

#evaluate & predeict
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test,verbose=0)
y_submit = model.predict(test_csv,verbose=0)

import datetime
dt = datetime.datetime.now()
submission_csv['count'] = y_submit
submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}.csv",index=False)

r2 = r2_score(y_test,y_predict)
print(f"{loss=}\n{r2=}")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
plt.title('따릉이')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(hist.history['loss'],label='loss',color='red',marker='.')
plt.plot(hist.history['val_loss'],label='val_loss',color='blue',marker='.')
plt.legend()
plt.show()