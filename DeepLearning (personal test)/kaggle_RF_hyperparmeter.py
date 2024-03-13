import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('C:\_data\kaggle\RF_hyperparameter\open//train.csv')

# person_id 컬럼 제거
X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']

# GridSearchCV를 위한 하이퍼파라미터 설정
param_search_space = {
    'n_estimators': [33],
    'max_depth': [None],
    'min_samples_split': [33],
    'min_samples_leaf': [33]
}

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=333)

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=rf, param_grid=param_search_space, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')

# GridSearchCV를 사용한 학습
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_params, best_score

submit = pd.read_csv('C:\_data\kaggle\RF_hyperparameter\open//sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('C:\_data\kaggle\RF_hyperparameter\open//baseline_submit1.csv', index=False)