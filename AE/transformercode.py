import numpy as np
import pandas as pd
from keras_transformer import get_model, decode
import sacrebleu
from datasets import load_dataset
from collections import defaultdict
data = load_dataset("embedding-data/altlex")

print(data.column_names)


source_sentences = data['train']['set'][:10000]
target_sentences = data['train']['set'][:10000]


print(source_sentences[0])
print(target_sentences[0])

# 각 문장을 문자 단위로 분리
source_tokens = [list(sentence) for sentence in source_sentences]
target_tokens = [list(sentence) for sentence in target_sentences]


def build_token_dict(token_list):
    # 초기에 명시적으로 특수 토큰 추가
    token_dict = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    return token_dict

# 딕셔너리 생성
source_token_dict = build_token_dict(source_tokens)
target_token_dict = build_token_dict(target_tokens)
target_token_dict_inv = {v: k for k, v in target_token_dict.items()}  # 역 딕셔너리 생성


def prepare_tokens(tokens, token_dict):
    max_len = max(map(len, tokens))
    token_dict = defaultdict(lambda: token_dict.get('<UNK>', 0), token_dict)  # handle unknown tokens
    return [list(map(lambda x: token_dict[x], t)) for t in tokens], max_len

source_input, source_max_len = prepare_tokens(source_tokens, source_token_dict)
target_input, target_max_len = prepare_tokens(target_tokens, target_token_dict)

# 디코딩 출력 준비
target_output = [[target_token_dict[t] for t in sentence] + [target_token_dict['<PAD>']] * 
                 (target_max_len - len(sentence)) for sentence in target_tokens]

# 모델 생성 및 컴파일
model = get_model(
    token_num=max(len(source_token_dict), len(target_token_dict)),
    embed_dim=128,
    encoder_num=4,
    decoder_num=3,
    head_num=8,
    hidden_dim=120,
    attention_activation='relu',
    feed_forward_activation='relu',
    dropout_rate=0.05,
    use_same_embed=False  # 다른 언어에 대해 다른 임베딩을 사용
)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

from keras.callbacks import EarlyStopping
# 모델 훈련
model.fit(
    x=[np.array(source_input), np.array(target_input)],
    y=np.array(target_output),
    epochs=10,
    batch_size=32,
    callbacks = [EarlyStopping(monitor='loss', patience=30, mode= 'min', verbose= 1, restore_best_weights=True)]
)

# 번역 예측
decoded = decode(
    model,
    source_input,
    start_token=source_token_dict['<START>'],
    end_token=source_token_dict['<END>'],
    pad_token=source_token_dict['<PAD>']
)

# 번역 결과 출력
for decoded_sequence in decoded:
    print(''.join([target_token_dict_inv[token] for token in decoded_sequence if token > 2]))  # 특수 토큰 제외하고 출력
    
    
    
#===================================

input_sentences = ["It is known as the Dairy Capital of Canada and promotes itself as `` The Friendly City."]
input_tokens = [list(sentence) for sentence in input_sentences]

# 토큰 추가 및 패딩
input_encoded, input_max_len = prepare_tokens(input_tokens, source_token_dict)

# 숫자 인코딩
input_encoded = [list(map(lambda x: source_token_dict.get(x, 0), tokens)) for tokens in input_encoded]

# 모델로 예측
input_decoded = decode(
    model,
    input_encoded,
    start_token=source_token_dict['<START>'],
    end_token=source_token_dict['<END>'],
    pad_token=source_token_dict['<PAD>']
)

# 번역 결과 출력 및 BLEU 점수 계산
for decoded_sequence in input_decoded:
    # 특수 토큰을 제외하고 출력
    translated_sentence = ''.join([target_token_dict_inv.get(token, '') for token in decoded_sequence if token > 2])
    print(translated_sentence)

    references = [["캐나다의 낙농 중심지로 알려져 있으며 '친절한 도시'로 자칭하고 있습니다.."]]
    hypothesis = [translated_sentence]  # Hypothesis should be a list of strings

    # BLEU 점수 계산
    bleu = sacrebleu.corpus_bleu(hypothesis, references)
    print(hypothesis, references)
    print(f"BLEU score: {bleu.score}")
