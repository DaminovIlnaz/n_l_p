from keras.preprocessing.text import Tokenizer
import pandas

train_df = pandas.read_csv('my_text.csv')

df = pandas.read_csv("all_text.csv", encoding="utf-8")
reviews = df['text'].values
tokenizer = Tokenizer()

max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)

text = [item for sublist in sequences for item in sublist]
vocab_size = len(tokenizer.word_index)
sentence_len = 15
pred_len = 5
train_len = sentence_len - pred_len
seq = []

for i in range(len(text)-sentence_len):
    seq.append(text[i:i+sentence_len])
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
trainX = []
trainy = []
for i in seq:
    trainX.append(i[:train_len])
    trainy.append(i[-1])
#reverse_word_map[1514]

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout

model = Sequential([
    Embedding(max_words, 50, input_length=train_len),
    LSTM(100, return_sequences=True),
    LSTM(100),
    Dense(100, activation='relu'),
    Dropout(0.1),
    Dense(max_words-1, activation='softmax')
])

from keras.callbacks import ModelCheckpoint
import numpy
import pandas as pd

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

filepath = "./model_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit(numpy.asarray(trainX),
         pd.get_dummies(numpy.asarray(trainy)),
         epochs = 50,
         batch_size = 128,
         callbacks = callbacks_list,
         verbose = 1)

from keras.preprocessing.sequence import pad_sequences
import numpy as np

def gen(model, seq, max_len=15):
    tokenized_sent = tokenizer.texts_to_sequences([seq])

    gen_res = seq.split(' ')

    while len(gen_res) < max_len:
        padded_sentence = pad_sequences(tokenized_sent[-10:], maxlen=10)
        op = model.predict(np.asarray(padded_sentence).reshape(1, -1))
        tokenized_sent[0].append(op.argmax() + 1)
        gen_res.append(reverse_word_map[op.argmax() + 1])

    return " ".join(gen_res)

my_test_sentences = [
    'я верю что одной из самых главных характеристик фильма является',
    'длиннющий пустой клип с красивыми картинками в котором зрителю отведена',
    'после некоторого количество переключений между историями я устал и мне ',
    'в визуальном плане кино бесспорно красивое и в этом его',
    'если честно я досмотрел фильм кое как с грустной миной',
    'по сути дела все от иллюзия иллюзия что жизнь это',
    'при всем при том что я очень люблю такие фильмы',
    'Фильм то себе так вовсе неплох он наделен и смыслом',
    'давно замечала что многие американские кинематографисты частенько перебарщивают с выжиманием',
    'итого получилась необычная история любви каких мир ещё не видел',
    'я начала смотреть этот фильм и меня не отпускала мысль',
    'можно было бы ещё многое сказать об этом мультике но',
    'помимо довольно посредственной и заезженной морали истории авторы мультфильма дает',
    'то что этот фильм кишит просторечными диалогами заметили все и',
    'если говорить о сценарии то это полный ноль после просмотра',
    'я не понял поступков главной героини и не понял того',
    'больше всего меня по ходу фильма раздражала как ни странно',
    'весь фильм мы видим только разные вариации поведения этого социопата',
    'начнем с того что император в древнем риме был чуть ли',
    'в общем получился очень тяжелый натужный фильм который не дает'
]

for i, value in enumerate(my_test_sentences, 1):
    print(i, gen(model, value))

