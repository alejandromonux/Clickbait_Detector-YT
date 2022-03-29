import json


def readFile(name):
    file = open(name, encoding="utf8")
    data = json.load(file)
    return data


def writeFile(name, dictionary):
    with open(name, "w") as file:
        json.dump(dictionary, file)
