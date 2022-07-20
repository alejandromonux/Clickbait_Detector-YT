import os

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from Classification.BERT.bert import bertPreprocessing, bertPreprocessingArray, bertencodingsArray
from Tools.files import writeFile, readFile

def arrayBERTPreprocessing(array,y):
    i = 0
    array_p = []
    for i in range(len(array)):
        # Limpiamos los strings
        if type(array[i]) is dict:
            element =array[i]["title"]
        else:
            element =array[i]
        array_p.append( textCleanupForBert(element))
    processedArray = bertPreprocessingArray(array_p)
    return bertencodingsArray(processedArray,array,y)


def databaseBERTCleanup(database):
    database = readFile(database)
    i = 0
    for item in database["list"]:
        # Limpiamos los strings
        title = textCleanupForBert(item["title"])
        desc = textCleanupForBert(item["description"])
        # reescribimos en la nueva DB
        database["list"][i]["title"] = title
        database["list"][i]["description"] = desc
        i += 1
    bertPreprocessing(database)

def databaseCleanup():
    # file = open("database.json")
    # database = json.load(file)
    database = readFile(os.getcwd() + "\database.json")
    i = 0
    for item in database["list"]:
        # Limpiamos los strings
        title = textCleanup(item["title"])
        desc = textCleanup(item["description"])
        # reescribimos en la nueva DB
        database["list"][i]["title"] = title
        database["list"][i]["description"] = desc
        i += 1
    # Miramos de poner todas las entrada de igual longitud
    # Buscamos la longitud máxima
    maxTitle = 0
    maxDesc = 0
    for item in database["list"]:
        if len(item["title"]) > maxTitle:
            maxTitle = len(item["title"])
        if len(item["description"]) > maxDesc:
            maxDesc = len(item["description"])
    # Reajustamos según longitud máxima
    j = 0
    for item in database["list"]:
        if len(item["title"]) < maxTitle:
            for i in range(len(item["title"]), maxTitle):
                item["title"].insert(i, "0")
        if len(item["description"]) < maxDesc:
            for i in range(len(item["description"]), maxDesc):
                item["description"].insert(i, "0")
        database["list"][j] = item
        j += 1

    # Escribim en un arxiu nou
    # with open("clean_database.json", "w") as file:
    #    json.dump(database,file)
    writeFile(os.getcwd() + "\clean_database.json", database)


def textCleanup(text):
    stop_words = stopwords.words('english')

    # Tokenizamos
    words = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    outWords = []
    # Bucle para quitar signos de puntuación y stop words
    i = 0
    for word in words:
        if word.isalnum():
            if word not in stop_words:
                # Lemmatization de verbos
                outWords.insert(i, lemmatizer.lemmatize(word, pos="v"))
                i += 1

    return outWords


def textCleanupForBert(text):
    stop_words = stopwords.words('english')
    outWords = []
    outArray = ""
    i = 0
    words = text.split(" ")
    for word in words:
        if word.isalnum():
            if word not in stop_words:
                # Lemmatization de verbos
                outWords.insert(i, word)
                outArray += word + " "
                i += 1
    outArray = outArray[:-1]
    return outArray

def textCleanupForVader(text):
    stop_words = stopwords.words('english')
    text = text.replace("[^a-zA-Z#]", " ")
    text.lower()
    # Tokenizamos
    words = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    outWords = []
    # Bucle para quitar signos de puntuación y stop words
    i = 0
    for word in words:
        if word.isalnum():
            if word not in stop_words:
                # Lemmatization de verbos
                outWords.insert(i, lemmatizer.lemmatize(word, pos="v"))
                i += 1
    try:
        outSentence = ""+outWords[0]
        i = 0
        for word in outWords:
            if i != 0:
                outSentence = outSentence+' '+word
            i+=1
    except Exception as e:
        return ""

    return outSentence