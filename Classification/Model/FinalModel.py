import os
import random

import numpy as np
import sklearn.neural_network
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import xgboost as xgb
from Classification.DecisionTree.TreeLogic import constructData, constructDataFromIndexes
from Classification.Model.UnionLayer import Union_ARCH
from Tools.Preprocessing import textCleanupForVader


class Model:
    umbral = 0.8 #Used for purposes of saving computational resources
    def __init__(self, database,indexes,**kwargs):
        willImport = kwargs.get('willImport', False) #We check if we need to import weights or train
        flask_direction =kwargs.get('flask_direction', None) #We check if another directory is needed
        self.umbral=kwargs.get('umbral', None) #We check for a previously defined limit
        if self.umbral == None:
            self.umbral=0.8
        random.shuffle(database["list"])
        #If we do not import, we  create and train everything from scratch
        if not willImport:
            indices = range(len(database["list"]))
            train_indices, test_indices = train_test_split(indices,  random_state=420, test_size=0.3)
            if len(indexes[0]) == 0:
                self.train_data_Titles,self.test_data_Titles,\
                self.train_data_Features,self.test_data_Features,\
                self.train_comments, self.test_comments = constructDataFromIndexes([train_indices,test_indices], onlytitles=True, merge=False,
                                                                                   adjusted_database=flask_direction)
            else:
                self.train_data_Titles, self.test_data_Titles, \
                self.train_data_Features, self.test_data_Features, \
                self.train_comments, self.test_comments = constructDataFromIndexes(indexes,
                                                                                   onlytitles=True, merge=False,
                                                                                   adjusted_database=flask_direction)

            from numpy import ndarray
            #self.train_data_Titles["y"] = ndarray.tolist(self.train_data_Titles["y"])
            #self.train_data_Features["y"] = ndarray.tolist(self.train_data_Features["y"])
            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(self.train_data_Titles["y"]),
                                                 y=self.train_data_Titles["y"])

            # Title

            self.titleClassifier = RandomForestClassifier(n_estimators=70,
                                                criterion="gini",
                                                class_weight={0: class_weights[0],
                                                              1: class_weights[1]},
                                                max_features="sqrt")
            # Feats
            self.featuresClassifier = xgb.XGBClassifier(objective='binary:hinge',
                                                        n_estimators=90,
                                                        seed=420,
                                                        tree_method="gpu_hist",
                                                        predictor="gpu_predictor",
                                                        booster="gbtree",
                                                        )
            # Vader
            self.sentimentAnalyzer = SentimentIntensityAnalyzer()

            #Final Layer
            #self.unionLayer = Union_ARCH()
            self.unionLayer = kwargs.get('union', None)
            if  self.unionLayer==None:
                self.unionLayer = sklearn.neural_network.MLPClassifier(solver='sgd',
                                                                   alpha=1e-3,
                                                                   activation='identity',
                                                                   hidden_layer_sizes=(20,40,20),
                                                                   random_state=1)
        else:
            #If we do import, we just create default classifiers
            self.titleClassifier = RandomForestClassifier()
            # Feats
            self.featuresClassifier = xgb.XGBClassifier()
            # Vader
            self.sentimentAnalyzer = SentimentIntensityAnalyzer()
            # Final Layer
            self.unionLayer = sklearn.neural_network.MLPClassifier()

    #We dump the models from this training
    def saveModel(self):
        import joblib
        joblib.dump(self.titleClassifier,"RandomForest.json")
        joblib.dump(self.featuresClassifier, "XGBoost.json")
        #self.featuresClassifier.save_model("XGBoost.json")
        joblib.dump(self.unionLayer,"UnionLayer.json")

    #We load the models from the last training
    def loadModel(self,**kwargs):
        routes= ["RandomForest.json","XGBoost.json","UnionLayer.json"]
        prefix = kwargs.get('prefix', "")
        for i in range(len(routes)): routes[i] = os.getcwd()+prefix+routes[i]
        import joblib
        self.titleClassifier = joblib.load(routes[0])
        # Feats
        self.featuresClassifier = joblib.load(routes[1])
        # Final Layer
        self.unionLayer = joblib.load(routes[2])

    #We train every part separately
    def fit(self):
        self.titleClassifier= self.titleClassifier.fit(self.train_data_Titles["x"],self.train_data_Titles["y"])
        self.featuresClassifier= self.featuresClassifier.fit(self.train_data_Features["x"], self.train_data_Features["y"])

        #self.titleClassifier.fit(self.train_data_Titles["x"], self.train_data_Titles["y"])
        data = []
        for i in range(len(self.train_data_Titles["x"])):
           data.append(self.getArrayForUnion(i))

        self.unionLayer.fit(data,self.train_data_Titles["y"])
        #Save Settings
        print("Training done")
        self.saveModel()

    #We construct the array for the MLP
    def getArrayForUnion(self, index):
        data = [self.train_data_Titles["x"][index]]
        data_f = [self.train_data_Features["x"][index]]
        y = self.titleClassifier.predict(data)
        y_proba = self.titleClassifier.predict_proba(data)
        y_feat = self.featuresClassifier.predict(data_f)
        y_feat_proba = self.featuresClassifier.predict_proba(data_f)
        neg = 0
        neu = 0
        pos = 0
        for item in self.train_comments[index]:
            output = self.sentimentAnalyzer.polarity_scores(textCleanupForVader(item["text"]))
            neg += output["neg"]
            neu += output["neu"]
            pos += output["pos"]
        return [y[0],y_proba[0][0],y_proba[0][0],y_feat[0],y_feat_proba[0][0],y_feat_proba[0][1],neg,neu,pos]


    #We predict a set of videos
    def predictList(self,list):
        preds = []
        for item in list:
            preds.append(self.predict(item))
        return preds

    #We predict a single video
    def predict(self,object):

        y       =self.titleClassifier.predict([object["title"]])
        y_proba =self.titleClassifier.predict_proba([object["title"]])
        if y_proba[0][y] <= self.umbral:
            #Features predict
            y_feat= self.featuresClassifier.predict([object["features"]])
            y_feat_proba= self.featuresClassifier.predict_proba([object["features"]])
            #Sentiment Analysis of comments
            neg = 0
            neu = 0
            pos = 0
            for item in object["comments"]:
                output = self.sentimentAnalyzer.polarity_scores(textCleanupForVader(item["text"]))
                neg+=output["neg"]
                neu += output["neu"]
                pos += output["pos"]
            #Final
            pred = self.unionLayer.predict([[y[0],y_proba[0][0],y_proba[0][1],y_feat[0],y_feat_proba[0][0],y_feat_proba[0][1],neg,neu,pos]])
        else:
            pred = y[0].item()
            return pred,y_proba,[0,0],0,0,0
        pred = pred[0].item()
        return pred,y_proba,y_feat_proba,neg,neu,pos

    #We test the model with a set of data already loaded inside it
    def test(self):
        y=[]
        import time
        t0 = time.time_ns()
        for i in range(len(self.test_data_Titles["x"])):
            pred = self.predict({"title":self.test_data_Titles["x"][i],
                          "features":self.test_data_Features["x"][i],
                          "comments":self.test_comments[i]})
            y.append(pred[0])
        t1 = time.time_ns()
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        print(classification_report(self.test_data_Titles["y"], y))
        cf_matrix = confusion_matrix(self.test_data_Titles["y"], y)
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

        return y,self.test_data_Titles["y"], (t1-t0)

    #We get the model's parameters
    def get_params(self,deep=True):
        from Tools.files import readFile
        return {"umbral" : self.umbral,"uinon" : self.unionLayer, "database": readFile(os.getcwd() + "\\adjusted_database.json"),
                "indexes": [[], []]}

    #We set necessary parameters
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)