from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
# import function_package as fp

# data
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요',
    '글쎄', '별로에요','생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요',
    '상현이 바보', '반장 잘생겼다', '욱이 또 잔다'
]

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]) # 0: 8, 1: 7

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7,
#  '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14,
#  '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20,
#  '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '상현이': 24, '바보': 25, '반장': 26,
#  '잘생겼다': 27, '욱이': 28, '또': 29, '잔다': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
x = pad_sequences(x)
print(x.shape) # (15, 5)

word_size = len(token.word_index)+1
print(word_size) #31
'''
# model
############## Enbedding 1 ###############
# model = Sequential()
# model.add(Embedding(input_dim=31, output_dim=10))#, input_length=5))
# model.add(LSTM(512, input_shape=(5,1), activation='relu'))
# model.add(Dense(256))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()
""" #
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, None, 10)          310
 embedding (Embedding)       (None, 5, 10)             310         #input_length=5 를 명시해줬을 때 
 
Embedding layer 의 input_shape = (batch_size, input_length), output_shape=(batch_size, input_length, output_dim)
param = input_dim * output_dim
"""
############## Enbedding 2 ###############

input_dim= 31 : 최적
input_dim= 20 : 단어사전 개수보다 작을 때 : 연산량이 줄음
input_dim= 40 : 단어사전 개수보다 클떄 : 연산량 늘음, 임의의 랜덤 임베딩 생성

model = Sequential()
model.add(Embedding(31, 10))#, input_length=5))
model.add(LSTM(512, input_shape=(5,1), activation='relu'))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x,labels,epochs=300,verbose=2)

# evaluate 
loss = model.evaluate(x,labels)

print(f"loss: {loss[0]}\nACC:  {loss[1]}")
# loss: 1.3276240551363117e-08
# ACC:  1.0
'''