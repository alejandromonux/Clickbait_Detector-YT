import numpy
import re
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict, Counter  # For word frequency
from numpy import sort


def bagOfWords():
    from Tools.files import readFile
    database = readFile("./DataRetrieval/clean_database.json")
    i=0
    vocabulary = []
    #Generem la Bag Of Words
    for item in database["list"]:
        for w in item["title"]:
            if w not in vocabulary:
                vocabulary.insert(i,w)
                i += 1
        #database["list"][1]["title"] = sort(list(set(item["title"])))
    # Ordenar la llista
    vocabulary = sorted(list(set(vocabulary)))

    ratio_de_1 = 0
    size = len(vocabulary)
    for item in database["list"]:
        words = item["title"]
        #cracio de vectors llargada de llista vocabulary
        bag_vector = numpy.zeros(len(vocabulary))
        for w in words:
            for i, word in enumerate(vocabulary):
                if word == w:
                    bag_vector[i] += 1
        #Càlcul de ratio de 1 i size total.
        ratio_de_1 += list(bag_vector).count(1)

    ratio_de_1 = ratio_de_1/size
    print("{0} en un len de \n{1}\n".format(ratio_de_1, size))
    return ratio_de_1, size

def word2vec():
    from Tools.files import readFile
    database = readFile("./DataRetrieval/clean_database.json")
    i = 0


def df(token, document):
    DF = {}
    i = 0
    for w in document:
        try:
            DF[w] += 1
        except:
            DF[w] = 0
            # DF[w].add(i)
    i += 1
    return DF[token]

def tf_df():
    from Tools.files import readFile
    database = readFile("./DataRetrieval/clean_database.json")
    i = 0
    #Doc Frequency con títulos
    DF = {}
    for item in database["list"]:
        for w in item["title"]:
            try:
                DF[w] += 1
            except:
                DF[w] = 0
                #DF[w].add(i)
        i+=1
    #print(DF)
    print(len(DF))
    #Doc Frequency con descripciones
    DF = {}
    for item in database["list"]:
        for w in item["description"]:
            try:
                DF[w] += 1
            except:
                DF[w] = 0
                # DF[w].add(i)
        i += 1
    #print(DF)
    print(len(DF))

    #TF-IDF a full
    tf_idf = {}
    doc = 0
    for item in database["list"]:
        tokens =item["description"]
        counter = Counter(tokens + item["title"])
        import numpy as np
        for token in np.unique(tokens):
            tf = counter[token] / len(np.unique(tokens))
            docf = df(token, tokens)
            idf = np.log(len(database["list"]) / (docf + 1))
            tf_idf[doc, token] = tf * idf
            doc +=1
    #print(tf_idf)
    print(len(tf_idf))