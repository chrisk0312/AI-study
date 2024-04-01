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
                'reg_lambda': 10.121476098960851}

model = XGBClassifier()
model.set_params(early_stopping_rounds=10,**xgb_params)
# model = RandomForestClassifier()

# fit & pred
model.fit(x_train,y_train,
          eval_set=[(x_train,y_train), (x_test,y_test)],
          verbose=1,
          eval_metric='logloss',
          )

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)

# Score:  0.9649122807017544
# ACC:  0.9649122807017544

'''================================================================'''
# print(type(model.feature_importances_))
feature_importances_list = list(model.feature_importances_)
feature_importances_list_sorted = sorted(feature_importances_list)
print(feature_importances_list_sorted)
drop_feature_idx_list = [feature_importances_list.index(feature) for feature in feature_importances_list_sorted] # 중요도가 낮은 column인덱스 부터 기재한 리스트
print(drop_feature_idx_list)

result_dict = {}
for i in range(len(drop_feature_idx_list)-1): # 1바퀴에는 1개, 마지막 바퀴에는 29개 지우기, len -1해준 이유는 30개 지우면 안되니까
    drop_idx = drop_feature_idx_list[:i+1] # +1 해준 이유는 첫바퀴에 0개가 아니라 1개를 지워야하니까
    new_x_train = np.delete(x_train,drop_idx,axis=1)
    new_x_test = np.delete(x_test,drop_idx,axis=1)
    print(new_x_train.shape,new_x_test.shape)
    
    model2 = XGBClassifier()
    model2.set_params(early_stopping_rounds=10,**xgb_params)
    model2.fit(new_x_train,y_train,
          eval_set=[(new_x_train,y_train), (new_x_test,y_test)],
          verbose=0,
          eval_metric='logloss',
          )
    new_result = model2.score(new_x_test,y_test)
    print(f"{i+1}개 컬럼이 삭제되었을 때 Score: ",new_result)
    result_dict[i+1] = new_result - result    # 그대로 보면 숫자가 비슷해서 구분하기 힘들기에 얼마나 변했는지 체크
    
    
print(result_dict)