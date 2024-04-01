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

x, y = fetch_covtype(return_X_y=True)
y = y-1

import warnings
warnings.filterwarnings('ignore')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    stratify=y
)

print(x_train.shape,y_train.shape)
print(np.unique(y_train,return_counts=True))


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

model = XGBClassifier()
model.set_params(early_stopping_rounds=10,**xgb_params)
# model = RandomForestClassifier()

# fit & pred
model.fit(x_train,y_train,
          eval_set=[(x_train,y_train), (x_test,y_test)],
          verbose=1,
          eval_metric='mlogloss',
          )

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)

# evaluate

# best param :  {'n_estimators': 400, 'min_child_weight': 0.01, 'max_depth': 6, 'learning_rate': 0.1, 'gamma': 0, 'early_stoppint_rounds': 50}
# ACC score  :  0.9527777777777777

'''================================================================'''
result = model.score(x_test,y_test)
feature_importances_list = list(model.feature_importances_)
feature_importances_list_sorted = sorted(feature_importances_list)
# print(feature_importances_list)
drop_feature_idx_list = [feature_importances_list.index(feature) for feature in feature_importances_list_sorted] # 중요도가 낮은 column인덱스 부터 기재한 리스트
print(drop_feature_idx_list)

result_dict = {}
for i in range(len(drop_feature_idx_list)-1): # 1바퀴에는 1개, 마지막 바퀴에는 29개 지우기, len -1해준 이유는 30개 지우면 안되니까
    drop_idx = drop_feature_idx_list[:i+1] # +1 해준 이유는 첫바퀴에 0개가 아니라 1개를 지워야하니까
    new_x_train = np.delete(x_train,drop_idx,axis=1)
    new_x_test = np.delete(x_test,drop_idx,axis=1)
    print(new_x_train.shape,new_x_test.shape)
    
    model.fit(new_x_train,y_train,
          eval_set=[(new_x_train,y_train), (new_x_test,y_test)],
          verbose=0,
          eval_metric='mlogloss',
          )
    new_result = model.score(new_x_test,y_test)
    print(f"{i+1}개 컬럼이 삭제되었을 때 Score: ",new_result)
    result_dict[i+1] = new_result - result    # 그대로 보면 숫자가 비슷해서 구분하기 힘들기에 얼마나 변했는지 체크
    
for data in result_dict.items():
    print(data)
# print(result_dict)

# {1: 0.000395858970938856, 2: 4.3028149015134076e-05, 3: 0.00020653511527246593, 4: 0.0013166613598616372, 5: 0.000938013648528857, 6: 0.0019620835950879822, 7: 0.0021169849315422207, 8: 0.0022116468593754712, 9: 0.001187576912816457, 10: 0.001101520614786189, 11: -0.001084309355180113, 12: -0.00011187318743921537, 13: -0.001729731590406458, 14: -0.00041307023054482084, 15: -0.00037004208152968676, 16: -0.0018588160374516383, 17: -0.0017985766288305394, 18: -0.011256163782346396, 19: -0.013880880872266577, 20: -0.012417923805753683, 21: -0.01947454024422779, 22: -0.01799437191810882, 23: -0.01913031505210705, 24: -0.015713880020309268, 25: -0.03308004096279782, 26: -0.03416435031797793, 27: -0.09765668700463837, 28: -0.14425617238797617, 29: -0.14463482009930895, 30: -0.14577936886311016, 31: -0.14620104472345807, 32: -0.14707021333356274, 33: -0.14718208652100195, 34: -0.15091692985551142, 35: -0.15122673252842, 36: -0.1514762957927076, 37: -0.15219056306635792, 38: -0.1528101684121752, 39: -0.15442802681514245, 40: -0.1548755195648993, 41: -0.15497018149273256, 42: -0.1548755195648993, 43: -0.16103715050385958, 44: -0.1650473739920656, 45: -0.16626937342409398, 46: -0.1683261189470151, 47: -0.16823145701918196, 48: -0.17529667908745894, 49: -0.17639819970224513, 50: -0.17953925458034647, 51: -0.18617419515847267, 52: -0.19496054318735312, 53: -0.1981274149548634}