# API client library
import sys
import os
import googleapiclient.discovery
import numpy
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

from Classification.BERT import bert
from Classification.BERT.bert import fineTuningBert
from Classification.Boosting.xgboost import XGBoost
from Classification.Clustering.k_means_clustering import kmeans_test, elbowmethod, elbowmethod_2
from Classification.DecisionTree.TreeClass import VideoInfo
from Classification.Model.FinalModel import Model
from Classification.RandomForest.randomForest import classifyWithForest, Ten_foldForest
from Classification.SentimentAnalysis.Vader import sentimentAnalysis
from DataRetrieval.Fix_database import dabasefix, databaseAdding, noRepeats, ReviseAllRatingsFromClass, \
    removeTheStrings, ratingsFromAll
from Encoding.encoding_test import bagOfWords, tf_df
from Tools.Preprocessing import textCleanup, textCleanupForBert, databaseCleanup, databaseBERTCleanup, \
    arrayBERTPreprocessing
from Tools.files import writeFile, readFile, readCSV, readCSVKaggle


def convertToFormat(db_name,rating):
    x,y= readCSVKaggle(db_name,rating)
    for i in range(len(x)):
        x[i]["comments"]=getVideoComments(x[i]["title"], x[i]["id"])
    db = {"x":x,"y":y}
    return db

def getVideoComments(name,id):
    # API information
    api_service_name = "youtube"
    api_version = "v3"
    # API key
    DEVELOPER_KEY = "AIzaSyAJ_jIg4eJM8hdWV9-6HXtB-60DoKzn4qc"
    # API client
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            maxResults=5,
            videoId=id
        )
        response = request.execute()
        for comment in response['items']:
            comments.append({"text": comment['snippet']['topLevelComment']['snippet']['textOriginal'],
                             "rating": 0})
    except Exception as e:
        print(e)
        print("problem with Video " + name + " / ID: " + id)
    return comments

def getChannelComments(name, id, database):
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
        part="snippet,contentDetails,statistics",
        id=id,
    )
    # Query execution
    responseChannel = request.execute()
    # Now with the response, we get the video list
    try:
        request = youtube.playlistItems().list(
            part="snippet",
            maxResults=50,
            playlistId=responseChannel['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        )
    except:
        print("problem with tuple " + name + "/" + id)
        return database

    response = request.execute()
    # Procesar el resultado y guardarlo en database
    index = 0
    for item in response['items']:
        for example in database["list"]:
            if item['snippet']['title'] == example["title"]:
                comments = getVideoComments(name=item['snippet']['title'],id=item['snippet']['resourceId']['videoId'])
                if len(comments)==0:
                    break
                else:
                    database["list"][database["list"].index(example)]["comments"] = comments
                    database["list"][database["list"].index(example)]["videoId"] = item['snippet']['resourceId']['videoId']
        index += 1

    return database

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
        part="snippet,contentDetails,statistics",
        id=id,
    )
    # Query execution
    responseChannel = request.execute()
    # Now with the response, we get the video list
    try:
        request = youtube.playlistItems().list(
            part="snippet",
            maxResults=25,
            playlistId=responseChannel['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        )
    except:
        print("problem with tuple " + name + "/" + id)
        return database

    response = request.execute()
    # Procesar el resultado y guardarlo en database
    index = 0
    for item in response['items']:
        try:
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=item['snippet']['resourceId']['videoId']
            )
        except:
            print("problem with Video " + item['snippet']['title'] + "/ID: " + item['snippet']['resourceId'])
            return database
        infoVideo = request.execute()
        title = item['snippet']['title']
        description = item['snippet']['description']
        try:
            database['list'].insert(
                index,
                {
                    "author": responseChannel["items"][0]["snippet"]["title"],
                    "subscribers": responseChannel["items"][0]["statistics"]["subscriberCount"],
                    "title": title,
                    "description": description,
                    "category": infoVideo['items'][0]['snippet']['categoryId'],
                    "publishDate": infoVideo['items'][0]['snippet']['publishedAt'],
                    "views": infoVideo['items'][0]['statistics']['viewCount'],
                    "likes": infoVideo['items'][0]['statistics']['likeCount'],
                    "fav_count": infoVideo['items'][0]['statistics']['favoriteCount'],
                    "num_comments": infoVideo['items'][0]['statistics']['commentCount'],
                    "rating": "0"
                }
            )
        except Exception as e:
            print(e)
            print("problem with Video " + item['snippet']['title'] + " from " + responseChannel["items"][0]["snippet"][
                "title"] + " -No Statistics")
        # database["list"][index]["title"] = title
        # database["list"][index]["description"] = description
        # database["list"][index]["rating"] = 0
        index += 1

    return database

def commentsLoop():
    channels = readFile(os.getcwd() + "\channels.json")
    database = readFile(os.getcwd() + "/adjusted_database.json")

    for i in channels["channels"]:
        name = i["name"]
        id = i["id"]
        print(name + "\n")
        database = getChannelComments(name, id, database)

    writeFile(os.getcwd() + "/adjusted_database.json", database)
    print("I hem acabat!")

def channelLoop():
    """
    file = open("channels.json")
    channels = json.load(file)
    """
    channels = readFile(os.getcwd() + "\channels.json")
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
    writeFile(os.getcwd() + "\database.json", database)

    print("I hem acabat!")

def databaseEncoding(db_option):
    bert.bertEncoding(db_option)

if __name__ == "__main__":
    option = sys.argv[1]
    # TODO: PROBAR A USAR EL READ Y WRITE DE FILES.PY EN VEZ DE LOS DEL MAIN Y VER CÓMO FUNCIONA PARA TEMA DIRECTORIOS
    option2 = sys.argv[2]
    option3 = sys.argv[3]
    # goToFunc(option)
    if option == '1':
        if option2 == '1':
            channelLoop()
        else:
            commentsLoop()
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
            kmeans_test(int(option3))
            ratingsFromAll()
        else:
            elbowmethod()
            elbowmethod_2()
    elif option == '5':
        ratingsFromAll()

        database = readFile(os.getcwd() + "\database.json")
        for item in database["list"]:
            print(item["title"] + "\n---\n" + str(item["rating"]) + "\n|||||||||")
    elif option == '6':
        fineTuningBert(os.getcwd() + "\\adjusted_database.json")
    elif option == '66':
        if option2 == '1':
            # Versión mala
            """
            channelLoop()
            databaseCleanup()
            databaseEncoding(option2)
            kmeans_test(3)
            ratingsFromAll()
            """
            #Temporalmente, versión de databaseFix
            a_db = readFile(os.getcwd() + "\\adjusted_database.json")
            db = readFile(os.getcwd() + "\database.json")
            dabasefix(a_db, db,int(option3), True)
            #contingencyPlan("logFile.txt",readFile(os.getcwd() + "\\database.json"))
            pass
        elif option2 == '2':
            # VERSIÓN BUENA
            # channelLoop()
            databaseBERTCleanup(os.getcwd() + "\database.json")
            databaseEncoding(option2)
            kmeans_test(int(option3))
            ratingsFromAll()
            # dabasefix(readFile(os.getcwd()+"\\adjusted_database.json"),readFile(os.getcwd()+"\database.json"))
        elif option2 == '3':
            databaseAdding(readFile(os.getcwd() + "\old_adjusted_database.json"),
                           readFile(os.getcwd() + "\\adjusted_database.json"))
            # noRepeats(readFile(os.getcwd()+"\\adjusted_database.json"))
        elif option2 == '4':
            noRepeats(readFile(os.getcwd() + "\\adjusted_database.json"))
        elif option2 == '5':
            ReviseAllRatingsFromClass(readFile(os.getcwd() + "\\adjusted_database.json"), rating=int(option3))
        else:
            removeTheStrings(readFile(os.getcwd() + "\\adjusted_database.json"))
    elif option == '7':
        databaseBERTCleanup(os.getcwd() + "\\adjusted_database.json")
        # From cleanup we go on to training
        fineTuningBert(os.getcwd() + "\\BERT_clean_database.json",int(option3))
    elif option == '8':
        from Classification.DecisionTree.TreeLogic import getBestFeatures, findAverage

        getBestFeatures()
        #findAverage(int(option3),False)
    elif option == '9':
        #databaseBERTCleanup(os.getcwd() + "\\adjusted_database.json")
        #databaseEncoding(option2)
        classifyWithForest()
        XGBoost()
    elif option == '10':
        Ten_foldForest()
    elif option == '11':
        sentimentAnalysis(readFile(os.getcwd() + "\\adjusted_database.json"))
    elif option == '12':
        model = Model(readFile(os.getcwd() + "\\adjusted_database.json"), [[],[]])
        model.fit()
        model.test()
    elif option == '13':
        database = readFile(os.getcwd() + "\\adjusted_database.json")
        indices = range(len(database["list"]))
        scores = []
        kfold = KFold(n_splits=10, shuffle=True, random_state=1)
        for train,test in kfold.split(database["list"]):
            model = Model(database, [train,test])
            model.fit()
            (preds,y) = model.test()
            score_f1 = sklearn.metrics.f1_score(y, preds)
            score_acc = sklearn.metrics.accuracy_score(y, preds)
            score_recall = sklearn.metrics.recall_score(y, preds)
            scores.append([score_f1, score_acc, score_recall])
        scores = numpy.array(scores)
        print('Cross Validation f1 scores: %s' % scores[:, 0])
        print('Cross Validation f1: %.3f +/- %.3f' % (np.mean(scores[:, 0]), np.std(scores[:, 0])))
        print('Cross Validation accuracy scores: %s' % scores[:, 1])
        print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores[:, 1]), np.std(scores[:, 1])))
        print('Cross Validation score_recall scores: %s' % scores[:, 2])
        print('Cross Validation score_recall: %.3f +/- %.3f' % (np.mean(scores[:, 2]), np.std(scores[:, 2])))

    elif option == '14':
        db_name = (os.getcwd()+"/encoded_other.json") if option2=="2" else (os.getcwd()+"/yt_other/encoded_5sc_db.json")
        model = Model(readFile(os.getcwd() + "\\adjusted_database.json"), [[], []])
        if not os.path.exists(db_name):
            if option2=="1":
                x,y=readCSV(os.getcwd()+"/clickbait_data.csv")
                db = arrayBERTPreprocessing(x, y)
                writeFile(os.getcwd() + "/encoded_other.json", db)
            else:
                db_pre=readFile(os.getcwd()+"/yt_other/5sc_db.json")
                x=db_pre["x"]
                y=db_pre["y"]
                db = arrayBERTPreprocessing(x, y)
                writeFile(os.getcwd() + "/yt_other/encoded_5sc_db.json", db)
            #databaseBERTCleanup(os.getcwd() + "\database.json")
        else:
            db=readFile(db_name)
        model.fit()
        if option2 == "1":
            preds = model.predictList(db["x"])
        else:
            preds = []
            for item in db["x"]:
                preds.append(model.predict(item))
        print(classification_report(preds, db["y"]))
        cf_matrix = confusion_matrix(preds, db["y"])
        print(cf_matrix)
        # We plot the matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ax.xaxis.set_ticklabels(['0', '1'])
        ax.yaxis.set_ticklabels(['0', '1'])
        # Display the visualization of the Confusion Matrix.
        plt.show()
        
        #TODO: Delete this
        othersDB = readFile(os.getcwd() + "/yt_other/other_db.json")
        for i in range(len(preds)):
            if preds[i] == 1 and db["y"][i] == 0:
                print(othersDB["x"][i]["title"])
    elif option == "15":
        
        clickbait= convertToFormat(os.getcwd()+"/yt_other/clickbait.csv",0)
        notClickbait=convertToFormat(os.getcwd()+"/yt_other/notClickbait.csv",1)
        for i in range(len(notClickbait["x"])):
            clickbait["x"].append(notClickbait["x"][i])
            clickbait["y"].append(notClickbait["y"][i])
        writeFile(os.getcwd()+"/yt_other/other_db.json",clickbait)
