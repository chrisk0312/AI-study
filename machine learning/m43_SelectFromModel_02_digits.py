from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np

x, y = load_digits(return_X_y=True)

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
                'reg_lambda': 10.121476098960851}

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
thresholds = np.sort(model.feature_importances_)
from sklearn.feature_selection import SelectFromModel

acc_dict = {}
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
    acc_dict[n] = round(new_result-result,6)
    
for data in acc_dict.items():
    print(data)
    
# (0, 0.0)
# (1, 0.0)
# (2, 0.0)
# (3, 0.0)
# (4, 0.0)
# (5, 0.002778)
# (6, 0.0)
# (7, 0.002778)
# (8, 0.002778)
# (9, 0.002778)
# (10, 0.002778)
# (11, 0.002778)
# (12, 0.002778)
# (13, 0.002778)
# (14, 0.002778)
# (15, 0.002778)
# (16, 0.0)
# (17, 0.005556)
# (18, 0.005556)
# (19, -0.002778)
# (20, 0.0)
# (21, 0.005556)
# (22, 0.002778)
# (23, 0.002778)
# (24, 0.008333)
# (25, -0.002778)
# (26, -0.002778)
# (27, 0.005556)
# (28, 0.002778)
# (29, 0.002778)
# (30, 0.005556)
# (31, -0.002778)
# (32, -0.005556)
# (33, -0.002778)
# (34, 0.0)
# (35, -0.008333)
# (36, -0.013889)
# (37, -0.011111)
# (38, -0.016667)
# (39, -0.013889)
# (40, -0.019444)
# (41, -0.013889)
# (42, -0.025)
# (43, -0.022222)
# (44, -0.027778)
# (45, -0.022222)
# (46, -0.027778)
# (47, -0.022222)
# (48, -0.027778)
# (49, -0.038889)
# (50, -0.027778)
# (51, -0.033333)
# (52, -0.033333)
# (53, -0.05)
# (54, -0.069444)
# (55, -0.075)
# (56, -0.091667)
# (57, -0.111111)
# (58, -0.186111)
# (59, -0.222222)
# (60, -0.325)
# (61, -0.441667)
# (62, -0.536111)
# (63, -0.727778)