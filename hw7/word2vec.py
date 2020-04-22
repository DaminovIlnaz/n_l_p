import pandas
import pymorphy2
from nltk.tokenize import RegexpTokenizer

morph = pymorphy2.MorphAnalyzer()
nltk = RegexpTokenizer(r'\w+')

# Все данные
data = pandas.read_csv("C:\\Users\\admin\\Downloads\\comments.csv", encoding="utf-8")

# тестовые данные
test = data["title"].isin(['Холоп', "Король Лев", "Джокер"])
test = data[test]
del test['title']

# учебные данные
data = data.loc[data['title'] != "Холоп"]
data = data.loc[data['title'] != "Король Лев"]
data = data.loc[data['title'] != "Джокер"]
del data['title']

print(data.head)

# Приводим в норм форму и чистим от лишнего тестовые данные
uniqueWords = []
i = 0
sentences = []
for comment in data.values:
    i += 1
    words = comment[0].split()
    wordsArr = []
    for word in words:
        wordNF = nltk.tokenize(morph.parse(word)[0].normal_form)
        if len(wordNF) == 1:
            wordNF = wordNF[0]
        else:
            wordNF = ""
        uniqueWords.append(wordNF)
        wordsArr.append(wordNF)
    sentences.append(wordsArr)

# Приводим в норм форму и чистим от лишнего учебные данные
i = 0
testSentences = []
for comment in test.values:
    i += 1
    words = comment[0].split()
    wordsArr = []
    for word in words:
        wordNF = nltk.tokenize(morph.parse(word)[0].normal_form)
        if len(wordNF) == 1:
            wordNF = wordNF[0]
        else:
            wordNF = ""
        uniqueWords.append(wordNF)
        wordsArr.append(wordNF)
    testSentences.append(wordsArr)

from gensim.models import Word2Vec

uniqueWords = set(uniqueWords)
model = Word2Vec(sentences=sentences, min_count=1, size=50)

print(model.wv.most_similar('я'))
print(model.wv.most_similar('он'))
print(model.wv.most_similar('фильм'))
print(model.wv.most_similar('оценка'))
print(model.wv.most_similar('актер'))


def getMiddleValue(arrays):
    finalArray = []
    for i in range(50):
        finalArray.append(int(0))
    for array in arrays:
        for j in range(len(array)):
            finalArray[j] += array[j]
    for value in finalArray:
        value = value / len(arrays)
    return list(finalArray)


print(model.wv['картина'])

columns = []
for i in range(50):
    columns.append(i)
X_train = pandas.DataFrame(columns=columns)
index = 0
for comment in sentences:
    index += 1
    arrays = []
    for word in comment:
        vector = model.wv[word]
        arrays.append(vector)
    X_train.loc[index] = getMiddleValue(arrays)

Y_train = data["label"]
Y_test = test["label"]

from sklearn.ensemble import RandomForestClassifier

randomForestCLF = RandomForestClassifier(max_depth=20, random_state=0).fit(X_train, Y_train)

columns = []
for i in range(50):
    columns.append(i)
X_test = pandas.DataFrame(columns=columns)
index = 0
for comment in testSentences:
    index += 1
    arrays = []
    for word in comment:
        if word in list(model.wv.vocab.keys()):
            vector = model.wv[word]
            arrays.append(vector)
        else:
            vector = []
            for i in range(50):
                vector.append(0)
            arrays.append(vector)
    X_test.loc[index] = list(getMiddleValue(arrays))

from sklearn.metrics import precision_recall_fscore_support

randomForestPredict = randomForestCLF.predict(X_test)
randomForestMetric = precision_recall_fscore_support(test["label"].values, randomForestPredict, average='weighted')

print(randomForestMetric)
