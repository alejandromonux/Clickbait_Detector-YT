import os

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from Classification.CommonFunctions import trainingAndTest
from Classification.DecisionTree.TreeLogic import constructData
import xgboost as xgb

def  XGBoost():
    #We create construct the data
    data,titles = constructData(os.getcwd() + "/adjusted_database.json", onlytitles=False, merge=False)
    #We split the data
    train_x, test_x, train_y, test_y = train_test_split(data["x"], data["y"],
                                                                        random_state=420,
                                                                        test_size=0.3)
    #We create the model with arbitrary settings
    classifier = xgb.XGBClassifier(objective='binary:hinge',
                                                n_estimators=90,
                                                seed=420,
                                                tree_method="gpu_hist",
                                                predictor="gpu_predictor",
                                                booster="gbtree",
                                                )
    #We train and trest the model
    trainingAndTest(classifier,train_x,train_y,test_x,test_y)