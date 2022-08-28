from sklearn.metrics import confusion_matrix, classification_report

#We get the indices?
def getIndices(database, split):
    out = []
    for i in range(len(database["list"])):
        for j in range(len(split)):
            if database["list"][i]["title"] == split[j][12:]:
                out.append(i)
    return out

#We train and test a model based on data passed to the function
def trainingAndTest(classifier, train_x,train_y,test_x,test_y):
    import time
    milliseconds = time.time_ns()
    classifier.fit(train_x, train_y)
    print("Time Train" + str((time.time_ns() - milliseconds)))
    milliseconds = time.time_ns()
    results_y = classifier.predict(test_x)
    #results_y = classifier.predict_proba(test_x)
    print("Time Test" + str((time.time_ns() - milliseconds)))
    print(classification_report(test_y, results_y))
    cf_matrix = confusion_matrix(test_y, results_y)
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
    return classifier

#We check  analyze and rewrite ratings
def CheckTheProbability(classifier, train_x,train_y,test_x,test_y,text,index):
    from Tools.files import readFile
    import os
    db = readFile(os.getcwd() + "/adjusted_database.json")

    results_y = classifier.predict(test_x)
    probability_y = classifier.predict_proba(test_x)
    for i in range(len(results_y)):
        if probability_y[i][results_y[i]] <= 0.80:
            print("%d --> %s --- [0-->%s, 1-->%s]" % (results_y[i], text[index[i]], str(probability_y[i][0]), str(probability_y[i][1])))
            print("Rating: "+ str( test_y[i]))
            doItAgain = 0
            while not doItAgain:
                try:
                    rating = int(input("No rating?"))
                    if (rating >= 2):
                        for k in range(len(db["list"])):
                            if db["list"][k]["title"] == text[index[i]]:
                                db["list"][k]["rating"]=rating
                        doItAgain = 0
                    else:
                        doItAgain = 1
                except:
                    doItAgain = 1

    from Tools.files import writeFile
    writeFile(os.getcwd() + "/adjusted_database.json", db)
