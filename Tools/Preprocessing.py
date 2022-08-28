import os
import random

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from Classification.BERT.bert import bertPreprocessing, bertPreprocessingArray, bertencodingsArray
from Tools.files import writeFile, readFile

#We undersample the database to a limit per class
def undersamplingDB(database, limit, classes):
    database_out = {"list":[]}
    count = [0]*classes
    random.shuffle(database["list"])
    for i in range(len(database["list"])):
        if count[database["list"][i]["rating"]] < limit:
            database_out["list"].append(database["list"][i])
            count[database["list"][i]["rating"]]+=1
    return database_out

#We preprocess a batch of elements from an array
def arrayBERTPreprocessing(array,y):
    i = 0
    array_p = []
    for i in range(len(array)):
        # Limpiamos los strings
        if type(array[i]) is dict:
            element =array[i]["title"]
        else:
            element =array[i]
        print(element)
        array_p.append( textCleanupForBert(element))
    processedArray = bertPreprocessingArray(array_p)
    return bertencodingsArray(processedArray,array,y)

#We cleanup a whole database for the bert Preprocessing
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

#We do a cleanup (preprocessing) of the text in "database.json"
def databaseCleanup(**kwargs):
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

#We cleanup (preprocess) a string of text
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

#The text Cleanup acts as the one from before but aimed at BERT
def textCleanupForBert(text):
    stop_words = stopwords.words('english')
    outWords = []
    outArray = ""
    i = 0
    words = text.split(" ")
    for word in words:
        if word.isalnum() or (not word.isalnum() and ".." in word):
            if word.lower() not in stop_words:
                # Lemmatization de verbos
                outWords.insert(i, word)
                outArray += word + " "
                i += 1
    outArray = outArray[:-1]
    return outArray

#The text Cleanup acts as the one from before but aimed at VADER
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

#The text Cleanup acts as the one from before but aimed at TFIDF
def textCleanupForTFIDF(text):
    pass