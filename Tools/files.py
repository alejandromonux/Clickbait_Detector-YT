import csv
import json
import numpy as np

def readCSVManual(name):
    x = []
    y = []
    file = open(name, encoding="utf8")
    line=file.readline()
    #Skippeamos la primera, que es la cabecera
    line = file.readline()
    while line != "":
        x.append(line.split(",")[0])
        y.append(int(line.split(",")[1]))
        line=file.readline()
    return x, y

def readCSV(name):
    x = []
    y = []
    with open(name, encoding='utf-8') as file:
        csvFile = csv.DictReader(file)
        for line in csvFile:
            x.append(line["headline"])
            y.append(line["clickbait"])
    return x,y

def readCSVKaggle(name,rating):
    x = []
    y = []
    with open(name, encoding='utf-8') as file:
        csvFile = csv.DictReader(file)
        for line in csvFile:
            x.append({"author":"",
                      "subscribers":"0",
                      "title": line["title"],
                      "description": "",
                      "category": "0",
                      "publishDate": "",
                      "views": line["total_views"],
                      "likes": "0",
                      "fav_count": "0",
                      "num_comments": "0",
                      "id":line["video_id"]})
            y.append(rating)
    return x,y

def readFile(name):
    file = open(name, encoding="utf8")
    data = json.load(file)
    return data


def writeFile(name, dictionary):
    with open(name, "w") as file:
        json.dump(dictionary, file)
