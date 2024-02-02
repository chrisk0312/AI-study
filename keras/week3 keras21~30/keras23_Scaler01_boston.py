import warnings
warnings.filterwarnings('ignore')
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

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

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=100)

scaler = MaxAbsScaler()


# scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = StandardScaler()

#We use scalers in machine learning to standardize or normalize 
# the range of independent variables or features of data

# MaxAbsScaler is used when you want to scale the features of your dataset
# so that the maximum absolute value of each feature in the data is 1.0. 
# It does not shift or center the data, and thus it does not destroy any sparsity.

# MinMaxScaler is used when you want to scale the features of your dataset 
# so that they lie in a given range, usually between 0 and 1, or 
# so that the maximum absolute value of each feature is scaled to unit size. 
# This can be useful in optimization algorithms that require features on a 
# similar scale, like gradient descent, or algorithms that use distance measures, like K-nearest neighbors (KNN).

# RobustScaler is used when you want to scale the features of your dataset to handle outliers.
# Unlike other scalers like MinMaxScaler and StandardScaler, RobustScaler uses statistics based on percentiles 
# (the interquartile range, to be specific) to scale the data, which makes it robust to outliers.

# StandardScaler is used when you want to scale the features of your dataset 
# so that they have a mean of 0 and a standard deviation of 1.
# This is also known as standardization.

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.min(x_test))
print(np.max(x_train))
print(np.max(x_test))



print(x_train)
print(y_train)
print(x_test)
print(y_test)

'''
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
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs=150, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x_test, y_test) 
print("로스 :", loss)
y_predict = model.predict(x_test)
results = model.predict(x)

from sklearn.metrics import r2_score
r2= r2_score(y_test, y_predict)
print ("R2 스코어 :",r2)


# 로스 : 33.07808303833008 (mse)
# R2 스코어 : 0.6575566714761335

# 로스 : 3.9236836433410645 (mae)
# R2 스코어 : 0.6746762919060152

# 로스 : 3.1488635540008545
# R2 스코어 : 0.7225222369283609

'''