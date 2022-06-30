import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from Classification.CommonFunctions import trainingAndTest
from Classification.DecisionTree.TreeLogic import constructData
import numpy as np



def classifyWithForest():
    data = constructData(os.getcwd() + "\database.json", onlytitles=True)
    train_x, test_x, train_y, test_y = train_test_split(data["x"], data["y"],
                                                                        random_state=420,
                                                                        test_size=0.3)
    class_weights = compute_class_weight(class_weight ='balanced',
                                         classes = np.unique(train_y),
                                         y =  train_y)
    print("Class Weights:", class_weights)
    classifier  = RandomForestClassifier(n_estimators=100,
                                         criterion="entropy",
                                         class_weight={0:class_weights[0],
                                                       1:class_weights[1]})
    trainingAndTest(classifier,train_x,train_y,test_x,test_y)