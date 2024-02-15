from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#1.데이터
datasets = load_breast_cancer()


x = datasets.data
y= datasets.target


print(np.unique(y, return_counts= True)) #(array([0, 1]), array([212, 357])
print(pd.Series(y).value_counts()) #다 똑가틈
print(pd.DataFrame(y).value_counts()) #다 똑가틈 (dataframe을 쓰면 메모리를 더 씀,행렬데이터는 dataframe만 가능)
print(pd.value_counts(y)) #다 똑가틈
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.68, shuffle= False, random_state= 334)

model = LinearSVC(C=100)    

model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print("model.score:", results)
y_predict = model.predict(x_test)
print(y_predict)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(x_test)
acc = accuracy_score(y_predict,y_test)
print("acc:", acc)  
print("r2:", r2)