import pandas
import pymorphy2
from nltk.tokenize import RegexpTokenizer
morph = pymorphy2.MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')

df = pandas.read_csv("C:\\Users\\admin\\Downloads\\comments.csv", encoding="utf-8")

valid = df["title"].isin(['Холоп', "Король Лев", "Джокер"])
test = df[valid]
del test['title']

df = df.loc[df['title'] != "Холоп"]
df = df.loc[df['title'] != "Король Лев"]
df = df.loc[df['title'] != "Джокер"]
del df['title']

print(df.head)

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
texts = []
for tuple in df.values:
    tupleIndex+=1
    texts.append(tuple[0])
    words = tuple[0].split()
    wordsList = []
    for word in words:
        wordNF = tokenizeWord(morph.parse(word)[0].normal_form)
        uniqWords.append(wordNF)
        wordsList.append(wordNF)
    sentences.append(wordsList)
tupleIndex = 0
for tuple in test.values:
    tupleIndex+=1
    texts.append(tuple[0])
    words = tuple[0].split()
    wordsList = []
    for word in words:
        wordNF = tokenizeWord(morph.parse(word)[0].normal_form)
        uniqWords.append(wordNF)
        wordsList.append(wordNF)
    sentences.append(wordsList)

from gensim import corpora
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

import gensim
NUM_TOPICS = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=15)
for topic in topics:
    print(topic[1])

from gensim.models import CoherenceModel

coherence_model_lda = CoherenceModel(model=ldamodel, texts=sentences, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)





