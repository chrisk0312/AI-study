from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import fetch_covtype
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

import warnings
warnings.filterwarnings(action='ignore')

#data
path = "C:\\_data\\DACON\\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  
test_csv = pd.read_csv(path+"test.csv",index_col=0)         
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'],axis=1) #count 를 드랍, axis=0은 행, axis=1은 열
y = train_csv['count']

# print(np.unique(y,return_counts=True)) # 회귀 
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)
print(x.shape,y.shape)
print(np.unique(y,return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    # stratify=y
)



sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)


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

model = XGBRFRegressor()
model.set_params(early_stopping_rounds=10,**xgb_params)
# model = RandomForestClassifier()

# fit & pred
model.fit(x_train,y_train,
          eval_set=[(x_train,y_train), (x_test,y_test)],
          verbose=1,
          eval_metric='rmse',
          )

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
# acc = accuracy_score(y_test,pred)
# print("ACC: ",acc)

# evaluate

# best param :  {'n_estimators': 400, 'min_child_weight': 0.01, 'max_depth': 6, 'learning_rate': 0.1, 'gamma': 0, 'early_stoppint_rounds': 50}
# ACC score  :  0.9527777777777777

'''================================================================'''
thresholds = np.sort(model.feature_importances_)
from sklearn.feature_selection import SelectFromModel

acc_dict = {}
for n, i in enumerate(thresholds):
    selection = SelectFromModel(model,threshold=i,prefit=False)
    print(x_train.shape)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(f"{i}\n{select_x_test.shape=}\n{x_test.shape=}\n=======")
    model2 = XGBRFRegressor()
    model2.set_params(early_stopping_rounds=10,**xgb_params)
    model2.fit(select_x_train,y_train,
          eval_set=[(select_x_train,y_train), (select_x_test,y_test)],
          verbose=0,
        #   eval_metric='logloss',
          )
    new_result = model2.score(select_x_test,y_test)
    print(f"{n}개 컬럼 삭제, threshold={i:.4f} R2: {new_result}")
    acc_dict[n] = round(new_result-result,6)
    
for data in acc_dict.items():
    print(data)
    
# (0, 0.0)
# (1, -0.001187)
# (2, -0.003515)
# (3, -0.003655)
# (4, -0.003674)
# (5, -0.016325)
# (6, -0.045391)
# (7, -0.033009)
# (8, -0.009722