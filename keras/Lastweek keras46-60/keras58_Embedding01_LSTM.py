from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

#1 data
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.', '한번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', ' 참 재밋네요','상헌이 천재',
    '반장 못생겼다', '욱이 또 잔다',
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])


token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
#{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, 
# '추천하고': 7, '싶은': 8, '영화입니다': 9, '한번': 10, '더': 11, 
# '보고': 12, '싶어요': 13, '글쎄별로에요': 14, '생각보다': 15, 
# '지루해요': 16, '연기가': 17, '어색해요': 18, '재미없어요': 19, 
# '재미없다': 20, '재밋네요': 21, '상헌이': 22, '바보': 23, '반장': 24, 
# '못생겼다': 25, '욱이': 26, '또': 27, '잔다': 28}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], 
# [7, 8, 9], # [10, 11, 12, 13], [14],
# [15, 16], [17, 18], # [19], 
# [2, 20], [1, 21], [22, 23], 
# [24, 25], # [26, 27, 28]]
print(type(x)) #<class 'list'>
x_padded = pad_sequences(x, padding ='pre', 
                         maxlen= 5,
                         truncating='pre')

print(x_padded)
print(x_padded.shape) #(15, 5)

x= x_padded
y= labels


# Model
model = Sequential()
model.add(LSTM(units=100, input_shape=(5,1)))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (assuming you have a training set)
# Replace 'y_train' with your actual target variable
model.fit(x, y, epochs=10, batch_size=1)

# Evaluate the model on the test set
loss = model.evaluate(x, y)
print("Loss:", loss[0])
print("Loss:", loss[0])


results = model.predict(x)
results= np.around(results)
print("Predictions:", results)
