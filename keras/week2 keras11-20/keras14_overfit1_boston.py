#09.1 카피
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

#현재 사이킷 런 버전 1.3.0(pip uninstall scikit-learn, scikit-learn-intelex, scikit-image)
#pip install scikit-learn==0.23.2
#pip install scikit-learn==1.1.3

datasets = load_boston()
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape) #506,13
print(y)
print(y.shape) #506

print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 
# 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']

print(datasets.DESCR)

# [실습]
# train_size 0.7이상, 0.9이하
# R2 0.62 이상 


#1 데이터
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.7, random_state=100)

print(x_train)
print(y_train)
print(x_test)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13)) #Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
model.add(Dense(30))
model.add(Dense(95))
model.add(Dense(120))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

#3 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train, epochs=50, batch_size=1,
          validation_split=0.2,
          )

#4. 평가,예측
loss = model.evaluate(x_test, y_test) 
print("로스 :", loss)
y_predict = model.predict(x_test)
results = model.predict(x)


r2= r2_score(y_test, y_predict)
print ("R2 스코어 :",r2)

def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)
print("RMSE:",rmse)

print("걸린시간:", round)
print("===========hist===========")
print(hist)
print("===========hist===========")
print(hist.history) #과제 리스트, 딕셔너리(key value), 튜플
print("===========hist===========")
print(hist.history['loss'])
print("===========hist===========")
print(hist.history['val_loss'])
print("===========hist===========")



plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],c='red',label = 'loss',marker=".")
plt.plot(hist.history['val_loss'], c='blue',label ='val_loss', marker='.')
plt.legend (loc='upper right')
plt.title('보스턴 loss')
plt.xlabel('에포')
plt.ylabel('로스')
plt.grid()
plt.show()


# 로스 : 33.07808303833008 (mse)
# R2 스코어 : 0.6575566714761335

# 로스 : 3.9236836433410645 (mae)
# R2 스코어 : 0.6746762919060152