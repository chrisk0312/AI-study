from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Flatten, Embedding
from sklearn.metrics import r2_score





#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요','글쎄',
    '별로에요', '생각보다 지루해요','연기가 어색해요',
    '재미없어요','너무 재미없다', '참 재밌네요.',
    '상헌이 바보', '반장 잘생겼다','욱이 또 잔다',    
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index) #{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10,
#'번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20,
# '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '상헌이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '욱이': 28, '또': 29, '잔다': 30} #단어사전의 갯수.

x = token.texts_to_sequences(docs)
print(x) # [[2, 3], [1, 4], [1, 5, 6], 
# [7, 8, 9], [10, 11, 12, 13, 14], [15],
# [16], [17, 18], [19, 20],
# [21], [2, 22], [1, 23],
# [24, 25], [26, 27], [28, 29, 30]]
print(type(x)) #<class 'list'>
#x = np.array(x ) #차원이 달라서 에러뜨는거 보여주기위해 씀.
#print(x) 

from keras.utils import pad_sequences
pad_x = pad_sequences(x, padding= 'pre', 
                      maxlen=5,  
                      truncating= 'pre' )
print(pad_x)
print(pad_x.shape) # (15, 5)

x = pad_x
y = labels


word_size = len(token.word_index) +1
print(word_size)

#2. 모델구성

model = Sequential()
##################임베딩1
#model.add(Embedding(input_dim=31, output_dim= 10, input_length=5)) #input_length=5)) #단어사전의 갯수 30(최대인풋= +1), output_dim= 아랫단에 가져다주는 수 ,input_length=데이터 1개의 길이 5
#embedding shape ^ input_dim * output_dim ^                                             Embedding만 인풋이 앞에 옴!
#임베딩 인풋=2차원, 아웃풋= 3차원/ input_length는 주지 않아도 자동으로 잡힘.(작게 넣으면(약수만 가능) 성능 떨어짐!!)

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 10)             310

#  lstm (LSTM)                 (None, 10)                840

#  dense (Dense)               (None, 512)               5632

#  dense_1 (Dense)             (None, 512)               262656

#  dense_2 (Dense)             (None, 256)               131328

#  dense_3 (Dense)             (None, 128)               32896

#  dense_4 (Dense)             (None, 64)                8256

#  dense_5 (Dense)             (None, 32)                2080

#  dense_6 (Dense)             (None, 16)                528

#  dense_7 (Dense)             (None, 1)                 17

# =================================================================
# Total params: 444543 (1.70 MB)
# Trainable params: 444543 (1.70 MB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

#model.add(Embedding(input_dim=31, output_dim= 10,))

###############Embedding2
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 10)          310

#  lstm (LSTM)                 (None, 10)                840

#  dense (Dense)               (None, 512)               5632

#  dense_1 (Dense)             (None, 512)               262656

#  dense_2 (Dense)             (None, 256)               131328

#  dense_3 (Dense)             (None, 128)               32896

#  dense_4 (Dense)             (None, 64)                8256

#  dense_5 (Dense)             (None, 32)                2080

#  dense_6 (Dense)             (None, 16)                528

#  dense_7 (Dense)             (None, 1)                 17

# =================================================================
# Total params: 444543 (1.70 MB)
# Trainable params: 444543 (1.70 MB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

#model.add(Embedding(input_dim=31, output_dim= 10,))
# input_dim=31 #디폴트
# input_dim=20 #단어사전의 갯수보다 작을때 데이터에서 무작위로 빼냄                         연산량이 줄어들고 성능저하가 옴
# input_dim=40 #단어사전의 갯수보다 클때   데이터에서 무작위로 뽑아 랜덤 임베딩 생성         연산량이 늘어나고 성능은 보장되지 않음. 
###############Embedding3
model.add(Embedding(31, 10)) #돌아감!

###############Embedding4
#model.add(Embedding(31, 10, 5)) #안돌아감!

###############Embedding5
#model.add(Embedding(31, 10, input_length=5)) #돌아감!


model.add(LSTM(units=10, input_shape=(5,1)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# #3. 컴파일, 훈련
# model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['acc'] )
# model.fit(x, y, epochs= 500, batch_size= 1, )

# #4. 평가, 예측
# result = model.evaluate(x,y,verbose=0)
# y_predict = model.predict(x)
# y_predict= np.around(y_predict)
# print('로스:', result[0])
# print('acc:', result[1])



# print('y의 예측값:', y_predict)