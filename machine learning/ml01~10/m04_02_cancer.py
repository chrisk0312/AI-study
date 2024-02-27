from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators

#1.데이터
datasets = load_breast_cancer()


x = datasets.data
y= datasets.target


print(np.unique(y, return_counts= True)) #(array([0, 1]), array([212, 357])
print(pd.Series(y).value_counts()) #다 똑가틈
print(pd.DataFrame(y).value_counts()) #다 똑가틈 (dataframe을 쓰면 메모리를 더 씀,행렬데이터는 dataframe만 가능)
print(pd.value_counts(y)) #다 똑가틈
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.68, shuffle= False, random_state= 334)

allAlgorithms = all_estimators(type_filter='classifier')

print("allAlogrithms :", allAlgorithms)
print("모델의 갯수 :", len(allAlgorithms))

for name, algorithm in allAlgorithms:
    try:
        # model
        model = algorithm()
        #fit
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(name, "의 정답률 = ", acc)
    except:
        # print(name, "은 없음!")
        continue