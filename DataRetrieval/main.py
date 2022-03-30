# API client library
import json
import sys

from BERT import bert
import nltk
from nltk.stem import *
from nltk.corpus import stopwords
import googleapiclient.discovery
from sklearn.cluster import KMeans

from BERT.bert import bertPreprocessing, fineTuningBert
from Clustering.k_means_clustering import kmeans_test, elbowmethod, elbowmethod_2
from DataRetrieval.Fix_database import dabasefix, databaseAdding, noRepeats
from Encoding.encoding_test import bagOfWords, tf_df
from Tools.files import writeFile, readFile
from Clustering import k_means_clustering
import os


def getVideos(name, id, database):
    # API information
    api_service_name = "youtube"
    api_version = "v3"
    # API key
    DEVELOPER_KEY = "AIzaSyAJ_jIg4eJM8hdWV9-6HXtB-60DoKzn4qc"
    # API client
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    # TODO: Diferenciar entre username e id (mirar guiones y tal)
    request = youtube.channels().list(
        part="contentDetails",
        id=id,
    )
    # Query execution
    response = request.execute()
    # Now with the response, we get the video list
    try:
        request = youtube.playlistItems().list(
            part="snippet",
            maxResults=25,
            playlistId=response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        )
    except:
        print("problem with tuple " + name + "/" + id)
        return database

    response = request.execute()
    # Procesar el resultado y guardarlo en database
    index = 0
    for item in response['items']:
        title = item['snippet']['title']
        description = item['snippet']['description']
        database['list'].insert(
            index,
            {
                "title": title,
                "description": description,
                "rating": "0"
            }
        )
        # database["list"][index]["title"] = title
        # database["list"][index]["description"] = description
        # database["list"][index]["rating"] = 0
        index += 1

    return database


def channelLoop():
    """
    file = open("channels.json")
    channels = json.load(file)
    """
    channels = readFile(os.getcwd()+"\channels.json")
    """
    database = {
        "list":[
            {"title" : "a",
             "description": "a",
             "rating" : "1"}
        ]
    }
    """
    database = {}
    database["list"] = []

    for i in channels["channels"]:
        name = i["name"]
        id = i["id"]
        print(name + "\n")
        database = getVideos(name, id, database)

    # with open("database.json", "w") as file:
    #    json.dump(database,file)
    writeFile(os.getcwd()+"\database.json", database)

    print("I hem acabat!")

def textCleanup(text):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    # Tokenizamos
    words = nltk.word_tokenize(text)
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
    i=0
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
    database = readFile(os.getcwd()+"\database.json")
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
    writeFile(os.getcwd()+"\clean_database.json", database)

def databaseEncoding(db_option):
    bert.bertEncoding(db_option)

def ratingsFromAll():
    e_database = readFile(os.getcwd()+"\encoded_database.json")
    database = readFile(os.getcwd()+"\database.json")
    i = 0
    for item in e_database["list"]:
        database["list"][i]["rating"] = item["rating"]
        i += 1
    writeFile(os.getcwd()+"\database.json",database)

if __name__ == "__main__":
    option = sys.argv[1]
    #TODO: PROBAR A USAR EL READ Y WRITE DE FILES.PY EN VEZ DE LOS DEL MAIN Y VER CÓMO FUNCIONA PARA TEMA DIRECTORIOS
    option2 = sys.argv[2]
    # goToFunc(option)
    if option == '1':
        channelLoop()
    elif option == '2':
        if option2 == '1':
            databaseCleanup()
        else:
            databaseBERTCleanup()
    elif option == '3':
        if option2 == '1':
            databaseEncoding(option2)
        elif option == '2':
            bagOfWords()
            tf_df()
        else:
            databaseEncoding(option2)
    elif option == '4':
        if option2 == '1':
            kmeans_test(3)
            ratingsFromAll()
        else:
            elbowmethod()
            elbowmethod_2()
    elif option == '5':
        ratingsFromAll()

        database = readFile(os.getcwd()+"\database.json")
        for item in database["list"]:
            print(item["title"]+ "\n---\n" +  str( item["rating"]) + "\n|||||||||")
    elif option == '6':
        fineTuningBert(os.getcwd()+"\\adjusted_database.json")
    elif option == '66':
        if option2 == '1':
            #Versión mala
            """
            channelLoop()
            databaseCleanup()
            databaseEncoding(option2)
            kmeans_test(3)
            ratingsFromAll()
            """
            pass
        elif option2 == '2':
            #VERSIÓN BUENA
            channelLoop()
            databaseBERTCleanup(os.getcwd()+"\database.json")
            databaseEncoding(option2)
            kmeans_test(3)
            ratingsFromAll()
            dabasefix(readFile(os.getcwd()+"\\adjusted_database.json"),readFile(os.getcwd()+"\database.json"))
        elif option2 == '3':
            databaseAdding(readFile(os.getcwd()+"\old_adjusted_database.json"), readFile(os.getcwd()+"\\adjusted_database.json"))
            #noRepeats(readFile(os.getcwd()+"\\adjusted_database.json"))
        else:
            noRepeats(readFile(os.getcwd()+"\\adjusted_database.json"))
    elif option == '7':
        databaseBERTCleanup(os.getcwd()+"\\adjusted_database.json")
        #From cleanup we go on to training
        fineTuningBert(os.getcwd()+"\BERT_clean_database.json")