import numpy as np
import hyperopt as hp
from hyperopt import *
# print(hy.__version__)   # 0.2.7

search_space = {
    'x1': hp.quniform('x1', -10, 10, 1),
    'x2': hp.quniform('x2', -15, 15, 1),}

'''
hp.quniform(label, low, high, q) : label에 대해 low부터 high까지 q 간격으로 검색공간설정
hp.uniform(label, low, high): low부터 high까지 정규분포 형태로 검색공간설정
hp.randint(label, upper): 0부터 upper까지 random한 정수값으로 검색공간설정
hp.loguniform(label low, high) exp(uniform(low,high))값을 반환하며,
    반환값의 log변환된 값은 정규분포 형태를 가지는 검색공간 설정
'''
def objective_function(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_val =  x1**2 - 20*x2
    
    return return_val

trial_val = Trials()

best = fmin(
    fn= objective_function, # 목적함수
    space= search_space,    # 탐색범위
    algo= tpe.suggest,      # 알고리즘, default
    max_evals= 20,          # 탐색횟수
    trials= trial_val,      
    rstate= np.random.default_rng(seed=10)  # random state
)

print(best)

print(trial_val.results)
print(trial_val.vals)


print('|   iter   |  target  |    x1    |    x2    |')
print('---------------------------------------------')
x1_list = trial_val.vals['x1']
x2_list = trial_val.vals['x2']
for idx, data in enumerate(trial_val.results):
    loss = data['loss']
    print(f'|{idx:^10}|{loss:^10}|{x1_list[idx]:^10}|{x2_list[idx]:^10}|')
    