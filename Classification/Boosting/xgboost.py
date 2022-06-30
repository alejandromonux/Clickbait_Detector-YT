import os

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from Classification.CommonFunctions import trainingAndTest
from Classification.DecisionTree.TreeLogic import constructData
import xgboost as xgb

def XGBoost():
    data = constructData(os.getcwd() + "\database.json", onlytitles=True)
    train_x, test_x, train_y, test_y = train_test_split(data["x"], data["y"],
                                                                        random_state=420,
                                                                        test_size=0.3)
    classifier = xgb.XGBClassifier(objective='binary:hinge',
                                   n_estimators=10,
                                  seed=420)
    trainingAndTest(classifier,train_x,train_y,test_x,test_y)