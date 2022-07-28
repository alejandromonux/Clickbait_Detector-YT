import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import compute_class_weight

from Classification.CommonFunctions import getIndices, trainingAndTest
from Classification.DecisionTree.TreeLogic import constructData
from Tools.files import readFile


def NaiveBayes():
    data,titles = constructData(os.getcwd() + "/adjusted_database.json", onlytitles=True, merge=False)
    train_x, test_x, train_y, test_y = train_test_split(data["x"], data["y"],
                                                                        random_state=420,
                                                                        test_size=0.3)
    test_indices=getIndices(readFile(os.getcwd() + "\encoded_database.json"), test_x)
    class_weights = compute_class_weight(class_weight ='balanced',
                                         classes = np.unique(train_y),
                                         y =  train_y)
    classifier = GaussianNB()
    classifier = trainingAndTest(classifier,train_x,train_y,test_x,test_y)