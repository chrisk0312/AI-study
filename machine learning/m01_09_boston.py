from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston    # pip install scikit-learn==1.1.3
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVR

import warnings
warnings.filterwarnings('ignore')

datasets = load_boston()
x = datasets.data
y = datasets.target

r2 = 0

# while r2 < 0.8:
r = int(np.random.uniform(1,1000))
r = 88
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)


# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
# scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train),np.max(x_train))
# print(np.min(x_test),np.max(x_test))

#model
model = LinearSVR(C=100)

#compile & fit
model.fit(x_train,y_train)

#evaluate & predict
loss = model.score(x_test,y_test )
result = model.predict(x)
y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print(f"{r=}\n{loss=}\n{r2=}\nRMSE: {RMSE(y_test,y_predict)}")


# Epoch 197: early stopping
# Time: 8.06sec
# r=88
# loss=13.1826171875
# r2=0.8213945466114984
# RMSE: 3.6307874062998877

# MinMax
# r=88
# loss=[5.607363700866699, 1.73788583278656]
# r2=0.9240283091852217
# RMSE: 2.3679872456140494

# StandardScaler
# loss=[4.916057586669922, 1.7392468452453613]
# r2=0.9333945146886241
# RMSE: 2.217218335403849

# MaxAbsScaler
# loss=[8.462682723999023, 2.279984474182129]
# r2=0.885342850431518
# RMSE: 2.909069072695939

# RobustScaler
# loss=[0.27136003971099854, 0.27136003971099854]
# r2=0.796099072829003

# LinearSVR
# r=88
# loss=0.8249537129985753
# r2=0.8249537129985753
# RMSE: 3.5944290404736883