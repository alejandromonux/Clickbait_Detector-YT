import os
from io import StringIO

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from Classification.DecisionTree import TreeClass
from Tools.files import readFile


def initializeTree() -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier()
    tree.criterion = "entropy"
    return tree


def constructData(database, onlytitles):
    data = {"x": [],
            "y": []}
    database = readFile(database)
    database_encoded = readFile(os.getcwd() + "\encoded_database.json")
    i = 0
    for example in database["list"]:
        videoinformation = TreeClass.VideoInfo(False, example["title"], database_encoded["list"][i]["title"],
                                               example["views"], example["likes"], example["subscribers"],
                                               example["category"])
        if not onlytitles: data["x"].append(videoinformation.returnAsArray())
        else: data["x"].append(videoinformation.encodedTitle)
        data["y"].append(example["rating"])
        i += 1
    return data


def getBestFeatures():
    tree = initializeTree()
    data = constructData(os.getcwd() + "\database.json")
    tree.fit(data["x"], data["y"])
    # TODO: encontrar las features y cuáles son mejores
    feat_importance = tree.tree_.compute_feature_importances(
        normalize=True)  # tree.feature_importances_#tree.tree_.compute_feature_importances(normalize=False)
    print("feat importance = " + str(feat_importance))
    out = StringIO()
    # if not os.path.exists(os.getcwd()+'/../DecisionTree/tree.dot'):
    #    os.makedirs(os.getcwd()+'/../DecisionTree/tree.dot')
    out = export_graphviz(tree, out_file=os.getcwd() + '/../DecisionTree/tree.dot')


def findAverage(num_classes, trim):
    data = constructData(os.getcwd() + "\database.json",onlytitles=False)
    index = 0
    trimmed_data = {"x": [],
                    "y": []}
    if trim is True:
        nums = []
        for index in range(0, num_classes):
            nums.append(data["y"].count(index))
        max_examples = min(nums)
        index = 0
        for item in data["x"]:
            if ((trimmed_data["y"].count(0) < max_examples) and (data["y"][index] == 0)) or \
                    ((trimmed_data["y"].count(1) < max_examples) and (data["y"][index] == 1)):
                trimmed_data["x"].append(item)
                trimmed_data["y"].append(data["y"][index])
            index += 1
        data = trimmed_data
    averages = []
    categories = []
    for i in range(0, num_classes):
        averages.append(TreeClass.VideoInfo(True, 0, 0, 0, 0, 0, 0))
        categories.append(
            {"1": 0, "2": 0, "10": 0, "15": 0, "17": 0, "19": 0, "20": 0, "22": 0, "23": 0, "24": 0, "25": 0, "26": 0,
             "27": 0, "28": 0, "29": 0})
        """
        averages = [TreeClass.VideoInfo(True,0,0,0,0,0,0),
                    TreeClass.VideoInfo(True,0,0,0,0,0,0),
                    TreeClass.VideoInfo(True,0,0,0,0,0,0),
                    TreeClass.VideoInfo(True,0,0,0,0,0,0),
                    TreeClass.VideoInfo(True,0,0,0,0,0,0)]
        categories = [{"1":0,"2":0,"10":0,"15":0,"17":0,"19":0,"20":0,"22":0,"23":0,"24":0,"25":0,"26":0,"27":0,"28":0,"29":0},
                      {"1":0,"2":0,"10":0,"15":0,"17":0,"19":0,"20":0,"22":0,"23":0,"24":0,"25":0,"26":0,"27":0,"28":0,"29":0},
                      {"1":0,"2":0,"10":0,"15":0,"17":0,"19":0,"20":0,"22":0,"23":0,"24":0,"25":0,"26":0,"27":0,"28":0,"29":0},
                      {"1":0,"2":0,"10":0,"15":0,"17":0,"19":0,"20":0,"22":0,"23":0,"24":0,"25":0,"26":0,"27":0,"28":0,"29":0},
                      {"1":0,"2":0,"10":0,"15":0,"17":0,"19":0,"20":0,"22":0,"23":0,"24":0,"25":0,"26":0,"27":0,"28":0,"29":0}]
        """
    index = 0
    for videoinformation in data["x"]:
        rating = data["y"][index]

        averages[rating].emojiCount += videoinformation[0]  # .emojiCount
        averages[rating].hasTypicalPhrase += videoinformation[1]  # .hasTypicalPhrase
        averages[rating].superlatives += videoinformation[2]  # .superlatives
        averages[rating].punctuationCount += videoinformation[3]  # .punctuationCount
        averages[rating].bigNum += videoinformation[4]  # .bigNum
        averages[rating].sensationalistAdjectives += videoinformation[5]  # .sensationalistAdjectives
        averages[rating].capitalLetterWordsCount += videoinformation[6]  # .capitalLetterWordsCount
        averages[rating].subsPerView += videoinformation[7]  # .subsPerView
        averages[rating].likes += videoinformation[8]  # .likes
        categories[rating][videoinformation[9]] += 1  # videoinformation.category] += 1
        index += 1
    for i in range(0, num_classes):
        averages[i].emojiCount = averages[rating].emojiCount / data["y"].count(i)
        averages[i].hasTypicalPhrase = averages[i].hasTypicalPhrase / data["y"].count(i)
        averages[i].superlatives = averages[i].superlatives / data["y"].count(i)
        averages[i].punctuationCount = averages[i].punctuationCount / data["y"].count(i)
        averages[i].bigNum = averages[i].bigNum / data["y"].count(i)
        averages[i].sensationalistAdjectives = averages[i].sensationalistAdjectives / data["y"].count(i)
        averages[i].capitalLetterWordsCount = averages[i].capitalLetterWordsCount / data["y"].count(i)
        averages[i].subsPerView = averages[i].subsPerView / data["y"].count(i)
        averages[i].likes = averages[i].likes / data["y"].count(i)
    for item in categories:
        from matplotlib import pyplot as plt
        figure = plt.figure()
        ax = figure.add_axes([0, 0, 1, 1])
        values = ["1", "2", "10", "15", "17", "19", "20", "22", "23", "24", "25", "26", "27", "28", "29"]
        ax.bar(values, [item["1"], item["2"], item["10"], item["15"], item["17"], item["19"], item["20"], item["22"],
                        item["23"], item["24"], item["25"], item["26"], item["27"], item["28"], item["28"]])
        plt.show()
    print("Ja tenim els numeros!")
    return 0
