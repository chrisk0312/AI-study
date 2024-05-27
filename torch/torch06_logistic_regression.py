import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("torch:", torch.__version__, "device:", DEVICE)
# torch: 2.3.0+cpu device: cpu


#1 data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle= True,
                                                    random_state=66, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE) #torch가 float형으로 받아들이기 때문에 float형으로 변환
x_test = torch.FloatTensor(x_test).to(DEVICE) #torch가 float형으로 받아들이기 때문에 float형으로 변환
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE) 

#2 model
# model = Sequential()
# model.add(Dense(1, input_dim=1))
# model = nn.Linear(1,1).to(device) #input_dim=1, output_dim=1,
#input dim=1 : 입력 데이터의 차원이 1, output dim=1 : 출력 데이터의 차원이 1
model = nn.Sequential(
    nn.Linear(30,64),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.Linear(16,7),
    nn.Linear(7,1),
    nn.Sigmoid()
).to(DEVICE)    


x = torch.Tensor(x)
#3 model.fit
# model.compile(loss='mse', optimizer='adam')
criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.1)


# model.fit(x,y, epochs=1000, batch_size=1)
def train(model, x, y, optimizer, criterion):
    # model.train() #학습 모드로 변경
    optimizer.zero_grad() #기울기 초기화 : 기울기가 누적되는 것을 방지
    output = model(x) #forward(순전파) 연산
    loss = criterion(output, y) #오차 계산
    loss.backward() #역전파 시작, 기울기 계산
    optimizer.step() #가중치 갱신, 역전파 끝
    return loss.item() #loss 값 반환 : tensor -> scalar, loss.data[0] -> loss.item()

epcohs=2000
for epoch in range(1,epcohs+1):
    loss = train(model, x, y, optimizer, criterion)
    print('Epoch {}/{} loss: {}'.format(epoch, epcohs, loss))

print("====================================")


#4 model.predict
# loss = model.evaluate(x,y)
def evvaluate(model,criterion, x, y):
    model.eval() #평가 모드로 변경
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y_predict, y)
    return loss2.item()

loss2 = evvaluate(model, criterion, x, y)
print("최종 loss : ", loss2)

# result  = model.predict([4])
x = torch.Tensor([[100]]).to(DEVICE)
result = model(x)

# 최종 loss :  6.6331299422017764e-06        
# 예측값 :  3.9948348999023438

from sklearn.metrics import accuracy_score

x_test = x_test.to(DEVICE)
y_predict = model(x_test).cpu().detach().numpy()

# y_predict = model(x_test).cpu().detach().numpy()
print(y_predict)
