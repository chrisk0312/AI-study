import numpy as np  
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

#1 데이터
x,y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=777)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimator':1000, 
              'learning_rate':0.1, 
              'max_depth':3, #트리의 최대 깊이
              'gamma':0,
              'min_child_weight':0,
              'subsample':0.4, #전체 데이터의 몇 %를 쓸 것인가 
              'colsample_bytree':0.8, 
              'colsample_bylevel':0.7,
              'reg_alpha':0, #L1 규제, 가중치의 절대값에 대한 페널티,
              'reg_lambda':1, #L2 규제, 가중치의 제곱에 대한 페널티
              'random_state':3377, 
              'verbose':0,
            }
 
#2 모델
model = XGBClassifier()
model.set_params(**parameters)


#3 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['logloss'], eval_set=[(x_train, y_train),(x_test, y_test)])

#4 평가
results = model.score(x_test,y_test)
print('model.score :', results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)

print(model.feature_importances_)

thresholds = np.sort(model.feature_importances_)
print(thresholds)
# [0.00161156 0.00731585 0.0073556  0.00914522 0.00946747 0.01028653
#  0.01072333 0.010909   0.01107938 0.01134863 0.01199021 0.01245722
#  0.01293163 0.0130031  0.01547301 0.02109379 0.02280835 0.02328049
#  0.02354662 0.03178831 0.03183511 0.03517428 0.03655624 0.04751698
#  0.05596447 0.07218913 0.09124162 0.09262265 0.10109623 0.158188  ]
print("=============================================")
from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    print(i,"\t", select_x_train.shape,"\t", select_x_test.shape)
    selection_model = XGBClassifier()
    selection_model.set_params(early_stopping_round=10, **parameters, eval_metric='logloss')
    
    selection_model.fit(select_x_train, y_train, eval_set=[(select_x_train, y_train),(select_x_test, y_test)])
    select_y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    print("Trech=%.3f, n=%d, Accuracy: %.2f%%" %(i, select_x_train.shape[1], score*100))
 