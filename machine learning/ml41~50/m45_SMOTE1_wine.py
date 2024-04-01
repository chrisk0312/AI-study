from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

dataset = load_wine()
x = dataset.data
y = dataset.target # dataset['target']도 같음

print(x.shape,y.shape)                  # (178, 13) (178,)
print(np.unique(y,return_counts=True))  # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# == print(pd.value_counts(y))

x = x[:-35]                 # 일부러 불균형하게 하기
y = y[:-35]
print(x.shape,y.shape)      # (143, 13) (143,)
print(pd.value_counts(y))   # 1    71 | 0    59 | 2     13

r = np.random.randint(1000)
r = 642
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=r,stratify=y)

############## smote ############## 
print("===== smote =====")
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=r)
x_train, y_train = smote.fit_resample(x_train,y_train)

print(pd.value_counts(y_train))

# model
model = Sequential()
model.add(Dense(128,input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# compile & fit
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss',mode='auto',patience=100,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1024, batch_size=16,validation_split=0.2,verbose=2,callbacks=[es])

# evaluate
print("x_test, y_test: ",x_test.shape,y_test.shape)

loss = model.evaluate(x_test,y_test)
y_predict = np.argmax(model.predict(x_test),axis=1)
f1 = f1_score(y_test,y_predict,average='weighted')

print(y_test.shape,y_predict.shape)  #(15,) (15,)
print(f"{r=}\nLOSS: {loss[0]}\nACC:  {loss[1]}\nF1:   {f1}")
model.save(f"./model_save/m01_wine/F1_{f1:.6f}.h5")
