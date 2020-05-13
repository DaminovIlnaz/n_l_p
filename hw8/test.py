import pandas

df = pandas.read_csv("C:\\Users\\admin\\Downloads\\comments.csv", encoding="utf-8")

# тестовые данные
valid = df["title"].isin(['Холоп', "Король Лев", "Джокер"])
test = df[valid]
del test['title']

# удалили наши данные(оставили данные для обучения)
df = df.loc[df['title'] != "Холоп"]
df = df.loc[df['title'] != "Король Лев"]
df = df.loc[df['title'] != "Джокер"]
del df['title']

df.head

def tokenizeWord(word):
    word = tokenizer.tokenize(word)
    if len(word) == 1:
        word = word[0]
    else:
        word = ""
    return word

import pymorphy2
from nltk.tokenize import RegexpTokenizer
morph = pymorphy2.MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')
uniqWords = []
tupleIndex = 0
sentences = []
for tuple in df.values:
    tupleIndex+=1
    words = tuple[0].split()
    wordsList = []
    for word in words:
        wordNF = tokenizeWord(morph.parse(word)[0].normal_form)
        uniqWords.append(wordNF)
        wordsList.append(wordNF)
    sentences.append(wordsList)
tupleIndex = 0
testSentences = []
for tuple in test.values:
    tupleIndex+=1
    words = tuple[0].split()
    wordsList = []
    for word in words:
        wordNF = tokenizeWord(morph.parse(word)[0].normal_form)
        uniqWords.append(wordNF)
        wordsList.append(wordNF)
    testSentences.append(wordsList)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

print(len(uniqWords))

tokenizer = Tokenizer(num_words=len(uniqWords))
tokenizer.fit_on_texts(sentences)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 1000

X_train = tokenizer.texts_to_sequences(sentences)
X_test = tokenizer.texts_to_sequences(testSentences)

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(vocab_size)

from keras.utils import to_categorical
y_train = df["label"]
y_test = test["label"]

num_classes = 3

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


import numpy as np
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            word, *vector = line.split()
            word = word.split("_")[0]
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

embedding_dim = 300
embedding_matrix = create_embedding_matrix('C:/Users/sante/Desktop/model.txt', tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(embedding_matrix.shape)
print(nonzero_elements)
nonzero_elements / vocab_size


from keras.layers import Dense, Dropout, Activation, Input, Embedding, Conv1D, GlobalMaxPool1D
from keras.models import Sequential
from keras import layers
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(Conv1D(128, 3))
model.add(Activation("relu"))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


from sklearn.metrics import classification_report
model.fit(X_train, y_train, epochs=10, batch_size=128)
predictions = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))

model.fit(X_train, y_train, epochs=20, batch_size=128)
predictions = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))

model.fit(X_train, y_train, epochs=30, batch_size=128)
predictions = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))

