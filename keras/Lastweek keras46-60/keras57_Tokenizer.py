from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


text = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index) #{'마구': 1, '진짜': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6}

print(token.word_counts) #OrderedDict([('나는', 1), ('진짜', 3), ('맛있는', 1), ('밥을', 1), ('마구', 4), ('먹었다', 1)])

x =  token.texts_to_sequences([text])
print(x) #[[3, 2, 2, 2, 4, 5, 1, 1, 1, 1, 6]]


#1 to_categorical에서 첫번째 0 빼기
# Subtract 1 from each index in the sequence
x = [[idx - 1 for idx in seq] for seq in x]


from keras.utils import to_categorical

x1 = to_categorical(x)
print(x1)
print(x1.shape) #(1, 12, 9)   #(1, 12, 8)

#2 사이킷런 원핫인코더

# Initialize OneHotEncoder
x = np.array(x).reshape(-1,1)
encoder = OneHotEncoder(sparse=False)

# Reshape and fit_transform the sequence
x_encoded = encoder.fit_transform(x)

print(x_encoded)  #[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
print(x_encoded.shape) #(12, 8)
'''
#3 판다스 갯더미

x= np.array(x).reshape(-1,1)

# Flatten the list of lists
x_flat = [item for sublist in x for item in sublist]

# Convert the flattened sequence to a DataFrame
df = pd.DataFrame({'Tokenized': x_flat})

# Use get_dummies to create dummy variables
df_dummies = pd.get_dummies(df['Tokenized'], prefix='Word')

# Concatenate the original DataFrame and the dummy variable DataFrame
df = pd.concat([df, df_dummies], axis=1)

print(df)
print(df.shape) #(12, 9)
'''




