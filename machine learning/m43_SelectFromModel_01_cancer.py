import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=47)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# model 
xgb_params = {'learning_rate': 0.13349839953884737,
                'n_estimators': 99,
                'max_depth': 8,
                'min_child_weight': 3.471164143831403e-06,
                'subsample': 0.6661302167437514,            #dropout 비슷
                'colsample_bytree': 0.9856906281904222,
                'gamma': 4.5485144879936555e-06,
                'reg_alpha': 0.014276113125688179,
                'reg_lambda': 10.121476098960851,
                # 'tree_method' : 'gpu_hist',
                # 'predictor' : 'gpu_predictor',
                }

model = XGBClassifier()
model.set_params(early_stopping_rounds=10,**xgb_params)
# # model = RandomForestClassifier()

# # fit & pred
model.fit(x_train,y_train,
          eval_set=[(x_train,y_train), (x_test,y_test)],
          verbose=1,
          eval_metric='logloss',
          )
# model = RandomForestClassifier()
# model.fit(x_train,y_train)

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)

# Score:  0.9649122807017544
# ACC:  0.9649122807017544
# model = RandomForestClassifier()
'''================================================================'''
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# [0.00164523 0.00254707 0.00280537 0.0029459  0.00349732 0.00467501
#  0.00548839 0.00576104 0.00580801 0.00599047 0.00624544 0.00675722
#  0.00942199 0.00963188 0.01055923 0.01060074 0.01123477 0.01166585
#  0.01230622 0.01353535 0.01607466 0.01743191 0.02694841 0.03252788
#  0.0489464  0.07311124 0.07470486 0.11009338 0.20865612 0.24838263]
from sklearn.feature_selection import SelectFromModel

acc_list = {}
for n, i in enumerate(thresholds):
    selection = SelectFromModel(model,threshold=i,prefit=False)
    print(x_train.shape)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(f"{i}\n{select_x_test.shape=}\n{x_test.shape=}\n=======")
    model2 = XGBClassifier()
    model2.set_params(early_stopping_rounds=10,**xgb_params)
    model2.fit(select_x_train,y_train,
          eval_set=[(select_x_train,y_train), (select_x_test,y_test)],
          verbose=0,
        #   eval_metric='logloss',
          )
    new_result = model2.score(select_x_test,y_test)
    print(f"{n}개 컬럼 삭제, threshold={i:.4f} ACC: {new_result}")
    acc_list[n] = (new_result-result)
    
print(acc_list)