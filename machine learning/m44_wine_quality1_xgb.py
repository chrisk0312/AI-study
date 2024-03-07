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
x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']

print(np.unique(y,return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)
print(np.unique(y,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=42,stratify=y)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# model
from xgboost import XGBClassifier
xgb_params = {'learning_rate': 0.13349839953884737,
                'n_estimators': 99,
                'max_depth': 8,
                'min_child_weight': 3.471164143831403e-06,
                'subsample': 0.6661302167437514,            #dropout 비슷
                'colsample_bytree': 0.9856906281904222,
                'gamma': 4.5485144879936555e-06,
                'reg_alpha': 0.014276113125688179,
                'reg_lambda': 10.121476098960851,
                # 'nthread' : 20,
                'tree_method' : 'gpu_hist',
                'predictor' : 'gpu_predictor',
                }
model = XGBClassifier()
model.set_params(early_stopping_rounds=10,**xgb_params)
# model = XGBClassifier(**xgb_params)

# fit & pred
model.fit(x_train,y_train,
          eval_set=[(x_train,y_train), (x_test,y_test)],
          verbose=1,
          eval_metric='mlogloss',
          )
# model.fit(x_train,y_train)
pred = model.predict(x_test)
result = model.score(x_test,y_test)

print(result)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)

# import optuna

# def objectiveXGB(trial):
#     param = {
#         'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
#         'max_depth' : trial.suggest_int('max_depth', 8, 16),
#         'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
#         'gamma' : trial.suggest_int('gamma', 1, 3),
#         'learning_rate' : 0.01,
#         'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
#         'nthread' : -1,
#         # 'tree_method' : 'gpu_hist',
#         # 'predictor' : 'gpu_predictor',
#         'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
#         'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
#         'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
#         'random_state' : 1127
#     }
    
#     # 학습 모델 생성
#     xgb_model = XGBClassifier(**xgb_params)
#     xgb_model.fit(x_train,y_train,
#           eval_set=[(x_train,y_train), (x_test,y_test)],
#           verbose=0,
#           eval_metric='mlogloss',
#           )
    
#     # 모델 성능 확인
#     score = accuracy_score(xgb_model.predict(x_test), y_test)
    
#     return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objectiveXGB, n_trials=200)

# model.fit(x_train,y_train,
#           eval_set=[(x_train,y_train), (x_test,y_test)],
#           verbose=0,
#           eval_metric='mlogloss',
#           )

# acc = model.score(x_test,y_test)

# print("optuna AAC: ",acc)
# best_params = study.best_params
# print(best_params)