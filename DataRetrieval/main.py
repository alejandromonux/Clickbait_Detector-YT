# API client library
import sys
import os
import googleapiclient.discovery
import numpy
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV

from Classification.BERT import bert
from Classification.BERT.bert import fineTuningBert
from Classification.Boosting.xgboost import XGBoost
from Classification.Clustering.k_means_clustering import kmeans_test, elbowmethod, elbowmethod_2
from Classification.DecisionTree.TreeClass import VideoInfo
from Classification.LogisticRegression.logisticRegression import logisticRegression
from Classification.Model.FinalModel import Model
from Classification.NaiveBayes.NaiveBayes import NaiveBayes
from Classification.RandomForest.randomForest import classifyWithForest, Ten_foldForest
from Classification.SentimentAnalysis.Vader import sentimentAnalysis
from DataRetrieval.Fix_database import dabasefix, databaseAdding, noRepeats, ReviseAllRatingsFromClass, \
    removeTheStrings, ratingsFromAll
from Encoding.encoding_test import bagOfWords, tf_df
from Tools.Preprocessing import textCleanup, textCleanupForBert, databaseCleanup, databaseBERTCleanup, \
    arrayBERTPreprocessing, textCleanupForTFIDF, undersamplingDB
from Tools.files import writeFile, readFile, readCSV, readCSVKaggle
from yt_api import *

def convertToFormat(db_name,rating):
    x,y= readCSVKaggle(db_name,rating)
    for i in range(len(x)):
        x[i]["comments"]=getVideoComments(x[i]["title"], x[i]["id"])
    db = {"x":x,"y":y}
    return db


def databaseEncoding(db_option):
    bert.bertEncoding(db_option)

if __name__ == "__main__":
    #THIS MAIN IS ONLY TO BE USED IN A DEVELOPMENT ENVIRONMENT, MEANT FOR CHECKING, TESTING AND DEVELOPING THE MODEL.
    #We get the three necessary options
    option = sys.argv[1] #General option
    option2 = sys.argv[2] #Specific sub-option
    option3 = sys.argv[3] #Parameter
    if option == '1':
        #We can either go to the channel loop or comments loop option, downloading one of the two options
        if option2 == '1':
            channelLoop()
        else:
            commentsLoop()
    elif option == '2':
        #Option used for cleanup, a normal one or one for BERT can eb used
        if option2 == '1':
            databaseCleanup()
        else:
            databaseBERTCleanup()
    elif option == '3':
        #Different types of encoding can be tested from here
        if option2 == '1':
            databaseEncoding(option2)
        elif option == '2':
            bagOfWords()
            tf_df()
        else:
            databaseEncoding(option2)
    elif option == '4':
        #KMEANS is made from this option
        if option2 == '1':
            kmeans_test(int(option3))
            ratingsFromAll()
        else:
            elbowmethod()
            elbowmethod_2()
    elif option == '5':
        #This option works to spread the ratings
        ratingsFromAll()

        database = readFile(os.getcwd() + "\database.json")
        for item in database["list"]:
            print(item["title"] + "\n---\n" + str(item["rating"]) + "\n|||||||||")
    elif option == '6':
        #DEPRECATED OPTION. KEPT FOR LEGACY PURPOSES.
        #We train the BERT Model 1.0
        fineTuningBert(os.getcwd() + "\\adjusted_database.json")
    elif option == '66':
        #Diverse sub options
        if option2 == '1':
            #We do a databaseFix
            a_db = readFile(os.getcwd() + "\\adjusted_database.json")
            db = readFile(os.getcwd() + "\webRequestLogs.json")
            dabasefix(a_db, db,int(option3), False,weblogs=True)
            #contingencyPlan("",readFile(os.getcwd() + "\\database.json"))
            pass
        elif option2 == '2':
            # we get the data, classify it with kmeans and spread the ratings
            channelLoop()
            databaseBERTCleanup(os.getcwd() + "\database.json")
            databaseEncoding(option2)
            kmeans_test(int(option3))
            ratingsFromAll()
            # dabasefix(readFile(os.getcwd()+"\\adjusted_database.json"),readFile(os.getcwd()+"\database.json"))
        elif option2 == '3':
            #We add DB info forom one to another
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
        #DEPRECATED
        #We cleanup data and train BERT model 1.0
        databaseBERTCleanup(os.getcwd() + "\\adjusted_database.json")
        # From cleanup we go on to training
        fineTuningBert(os.getcwd() + "\\BERT_clean_database.json",int(option3))
    elif option == '8':
        #We search for the best features from the DB
        from Classification.DecisionTree.TreeLogic import getBestFeatures, findAverage

        getBestFeatures()
        #findAverage(int(option3),False)
    elif option == '9':
        #We cleanup, encode and test different models
        databaseBERTCleanup(os.getcwd() + "\\adjusted_database.json")
        databaseEncoding(option2)
        NaiveBayes()
        classifyWithForest()
        XGBoost()
    elif option == '10':
        Ten_foldForest()
    elif option == '11':
        sentimentAnalysis(readFile(os.getcwd() + "\\adjusted_database.json"))
    elif option == '12':
        #We test our model
        db = readFile(os.getcwd() + "\\adjusted_database.json")
        db_us= undersamplingDB(database=db,limit=1500,classes=2)
        model = Model(db_us, [[],[]])
        model.fit()
        y,y_true,time= model.test()
        print(time)
    elif option == '13':
        #We test our model with 10-Fold CV
        database = readFile(os.getcwd() + "\\adjusted_database.json")
        db_us= undersamplingDB(database=database,limit=1500,classes=2)
        indices = range(len(database["list"]))
        scores = []
        kfold = KFold(n_splits=10, shuffle=True, random_state=1)
        for train,test in kfold.split(database["list"]):
            model = Model(database, [train,test])
            model.fit()
            preds,y,time= model.test()
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
        #We test our model against other DBs
        db_name = (os.getcwd()+"/yt_other/encoded_5sc_db.json") if option2=="2" else (os.getcwd()+"/yt_other/encoded_other_dbf.json")
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
        """
        othersDB = readFile(os.getcwd() + "/yt_other/other_db.json")
        for i in range(len(preds)):
            if preds[i] == 1 and db["y"][i] == 0:
                print(othersDB["x"][i]["title"])
        """
    elif option == "15":
        #We test our model against other DBs
        clickbait= convertToFormat(os.getcwd()+"/yt_other/clickbait.csv",0)
        notClickbait=convertToFormat(os.getcwd()+"/yt_other/notClickbait.csv",1)
        for i in range(len(notClickbait["x"])):
            clickbait["x"].append(notClickbait["x"][i])
            clickbait["y"].append(notClickbait["y"][i])
        writeFile(os.getcwd()+"/yt_other/other_db.json",clickbait)
    elif option =="gridsearch":
        from sklearn.naive_bayes import GaussianNB
        model = Model(readFile(os.getcwd() + "\\adjusted_database.json"), [[], []])
        gscv = GridSearchCV(model, param_grid= {"umbral":[np.linspace(0,1,11)],
                                                "union":[sklearn.neural_network.MLPClassifier(solver='sgd',
                                                                       alpha=1e-3,
                                                                       activation='relu',
                                                                       hidden_layer_sizes=(120,60,30),
                                                                       random_state=1),
                                                        GaussianNB()],
                                                "database":[readFile(os.getcwd() + "\\adjusted_database.json")],
                                                "indexes":   [[[], []]]},
                            scoring=dict)
        gscv.fit([],[])

        print(sorted(gscv.cv_results_.keys()))
    elif option == "logres":
        #DEPRECATED We test logres
        #classifyWithForest()
        pass
    elif option == "validarUmbral":
        #We validate the threshold graphically
        times = []
        for i in np.arange(0,1, 0.1):
            model = Model(readFile(os.getcwd() + "\\adjusted_database.json"), [[], []],
                          umbral=i)
            model.fit()
            y,y_def,time = model.test()
            times.append(time)
        import matplotlib.pyplot as plt
        import numpy as np
        ypoints = np.array(times)
        plt.plot(ypoints, linestyle='solid')
        plt.show()
    elif option == "DEBUGWEB":
        #Used to debug Web methods
        dataForTheWeb("3D_I1M0dfFeck")
    elif option == "TFIDF":
        #Performance test of TFIDF
        db = readFile(os.getcwd()+"\\adjusted_database.json")
        db_clean = db
        for i in range(len(db["list"])):
            db_clean["list"][i]["title"] = textCleanupForBert(db["list"][i]["title"])
        titles = []
        for i in db_clean["list"]: titles.append(i["title"])
        from Encoding.tfidf import *
        titles = tfidf_encoding(titles)
        for i in range(len(titles)): db["list"][i]["title"]=titles[i].tolist()
        writeFile(os.getcwd()+"\\encoded_database.json",db_clean)
    elif option=="PASSTHEMTOTHEMODEL":
        #Test of the Request Logs through the model
        db = readFile(os.getcwd()+"\\webRequestLogs.json")
        model = Model(readFile(os.getcwd() + "\\adjusted_database.json"), [[], []], willImport=True)
        model.loadModel(prefix="\\")
        for i in range(len(db["list"])):
            predictionObject = arrayBERTPreprocessing([db["list"][i]], [0])
            output=model.predict(predictionObject["x"][0])
            db["list"][i]["rating"]= output[0]
        writeFile(os.getcwd()+"\\webRequestLogs.json", db)
    elif option=="USAGESTATS":
        #Usage Stats analysis
        y = []
        y_true = []
        db = readFile(os.getcwd() + "\\webRequestLogs.json")
        a_db = readFile(os.getcwd() + "\\adjusted_database.json")
        for item in db["list"]:
            for item_adb in a_db["list"]:
                if item["title"]==item_adb["title"]:
                    y.append(item["rating"])
                    y_true.append(item_adb["rating"])
        print(classification_report(y, y_true))
        cf_matrix = confusion_matrix(y, y_true)
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

    elif option=="USERSTATS":
        # User Stats analysis
        y = []
        y_true = []
        db = readFile(os.getcwd() + "\\webRequestLogs.json")
        for item in db["list"]:
            try:
                y.append(item["opinion"])
                y_true.append(item["rating"])
            except:
                print("Video doesn't have opinion")
        print(classification_report(y, y_true))
        cf_matrix = confusion_matrix(y, y_true)
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
