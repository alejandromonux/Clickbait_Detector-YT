import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight

from Classification.CommonFunctions import *
from Classification.DecisionTree.TreeLogic import constructData
import numpy as np

from Tools.files import readFile


def Ten_foldForest():
    import numpy
    import sklearn
    data = constructData(os.getcwd() + "/adjusted_database.json", onlytitles=False, merge=True)
    kfold = KFold(n_splits=10,shuffle=True,random_state=1)
    data_x = numpy.array(data["x"])
    data_y = numpy.array(data["y"])
    #scores = numpy.array([])
    scores = []
    for train,test in kfold.split(data_x):
        train_x,train_y,test_x,test_y = (data_x[train],data_y[train],data_x[test],data_y[test])
        #We train
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(train_y),
                                             y=train_y)
        classifier = RandomForestClassifier(n_estimators=100,
                                            criterion="entropy",
                                            class_weight={0: class_weights[0],
                                                          1: class_weights[1]})
        import time
        milliseconds = time.time_ns()
        classifier.fit(train_x,train_y)
        print("Time " + str((time.time_ns()-milliseconds)))
        preds = classifier.predict(test_x)
        #Calcular la score
        score_f1 = sklearn.metrics.f1_score(test_y,preds)
        score_acc = sklearn.metrics.accuracy_score(test_y,preds)
        score_recall = sklearn.metrics.recall_score(test_y,preds)
        scores.append([score_f1,score_acc,score_recall])
    scores = numpy.array(scores)
    print('Cross Validation f1 scores: %s' % scores[:,0])
    print('Cross Validation f1: %.3f +/- %.3f' % (np.mean(scores[:,0]), np.std(scores[:,0])))
    print('Cross Validation accuracy scores: %s' % scores[:,1])
    print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores[:,1]), np.std(scores[:,1])))
    print('Cross Validation score_recall scores: %s' % scores[:,2])
    print('Cross Validation score_recall: %.3f +/- %.3f' % (np.mean(scores[:,2]), np.std(scores[:,2])))

def getIndices(database, split):
    out = []
    for i in range(len(database["list"])):
        for j in range(len(split)):
            if database["list"][i]["title"] == split[j][12:]:
                out.append(i)
    return out

def classifyWithForest():
    data,titles = constructData(os.getcwd() + "/adjusted_database.json", onlytitles=True, merge=False)
    train_x, test_x, train_y, test_y = train_test_split(data["x"], data["y"],
                                                                        random_state=420,
                                                                        test_size=0.3)
    test_indices=getIndices(readFile(os.getcwd() + "\encoded_database.json"), test_x)
    class_weights = compute_class_weight(class_weight ='balanced',
                                         classes = np.unique(train_y),
                                         y =  train_y)
    print("Class Weights:", class_weights)
    classifier  = RandomForestClassifier(n_estimators=100,
                                         criterion="entropy",
                                         class_weight={0:class_weights[0],
                                                       1:class_weights[1]})

    classifier = trainingAndTest(classifier,train_x,train_y,test_x,test_y)
    #CheckTheProbability(classifier, train_x, train_y, test_x, test_y,titles, test_indices)