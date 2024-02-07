from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical
from sklearn.preprocessing import MaxAbsScaler

#1
path= "c:\_data\dacon\dechul\\"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
sample_csv=pd.read_csv(path+"sample_submission.csv")
x= train_csv.drop(['대출등급'],axis=1)
y= train_csv['대출등급']

print(np.unique(y,return_counts=True))


y = y.values.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(y).toarray()

lb=LabelEncoder()
lb.fit(x['대출기간'])
x['대출기간'] = lb.transform(x['대출기간'])
lb.fit(x['근로기간'])
x['근로기간'] = lb.transform(x['근로기간'])
lb.fit(x['주택소유상태'])
x['주택소유상태'] = lb.transform(x['주택소유상태'])
lb.fit(x['대출목적'])
x['대출목적'] = lb.transform(x['대출목적'])

lb.fit(test_csv['대출기간'])
test_csv['대출기간'] =lb.transform(test_csv['대출기간'])

lb.fit(test_csv['근로기간'])
test_csv['근로기간'] =lb.transform(test_csv['근로기간'])

lb.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] =lb.transform(test_csv['주택소유상태'])

lb.fit(test_csv['대출목적'])
test_csv['대출목적'] =lb.transform(test_csv['대출목적'])

x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.8,random_state=40,stratify=y_ohe)




scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.min(x_test))
print(np.max(x_train))
print(np.max(x_test))




def create_model(optimizer='adam'):
  model = Sequential()
  model.add(Dense(13, input_shape=(13,), activation='relu'))
  model.add(Dense(26))
  model.add(Dense(39))
  model.add(Dense(78))
  model.add(Dense(50))
  model.add(Dense(7, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
optimizers = ['rmsprop', 'adam']
param_grid = dict(optimizer=optimizers)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_macro', cv=3)
grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

sample_csv.to_csv(path + "sample_submission_3.csv", index=False)

