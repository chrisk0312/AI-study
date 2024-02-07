from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence  # Corrected import statement
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences

text = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다'
text2 = '상헌이가 선생을 괴롭힌다. 상헌이는 못생겼다. 상헌이는 마구 마구 못생겼다'

# Combine texts into a list
texts = [text, text2]

# Initialize Tokenizer
token = Tokenizer()
token.fit_on_texts(texts)

# Print word index and word counts for each text
for idx, txt in enumerate(texts, start=1):
    print(f"Text {idx} Word Index: {token.word_index}")
    print(f"Text {idx} Word Counts: {token.word_counts}")

# Convert texts to sequences
sequences = [token.texts_to_sequences([txt])[0] for txt in texts]

# Subtract 1 from each index in the sequence
sequences = [[idx - 1 for idx in seq] for seq in sequences]

from keras.utils import to_categorical

# to_categorical for each sequence
x1_list = [to_categorical(seq) for seq in sequences]

# Pad sequences to the same length
max_len = max(len(seq) for seq in sequences)
x1_padded = [sequence.pad_sequences(seq, maxlen=max_len) for seq in x1_list]

# Concatenate along the last axis (axis=-1)
x1_concatenated = np.concatenate(x1_padded, axis=-1)

print(x1_concatenated)
print(x1_concatenated.shape)

# Initialize OneHotEncoder
x_encoded = np.array(sequences).reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)

# Reshape and fit_transform the sequence
x_encoded = encoder.fit_transform(x_encoded)

print(x_encoded)
print(x_encoded.shape)