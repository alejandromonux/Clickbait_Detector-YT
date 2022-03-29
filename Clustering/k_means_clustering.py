import matplotlib
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from Tools.files import writeFile


def getTitles(database):
    out = []
    i = 0
    for item in database["list"]:
        out.insert(i,item["title"])
        i += 1
    return out

def plotTheClasses(graphic_bars):
    figure = plt.figure()
    ax = figure.add_axes([0, 0, 1, 1])
    values = ['0', '1', '2']
    ax.bar(values, graphic_bars)
    plt.show()

def plotTheElbow(y_values, x_value,title):
    plt.plot(x_value, y_values)
    plt.xlabel("Valor para K")
    plt.ylabel("SSE")
    plt.title(title)
    plt.show()

def kmeans_test(k_clusters):
    from Tools.files import readFile
    database = readFile("./DataRetrieval/encoded_database.json")
    array_of_titles = getTitles(database)
    kmeans = KMeans(init="random",
                    n_clusters=k_clusters,
                    n_init=10,
                    max_iter=300,
                    random_state=42)
    kmeans.fit(array_of_titles)
    print("No tenemos lost ark niño, sólo ffxiv")
    labels = kmeans.labels_
    labels = labels.tolist()
    graphic_bars = [
        labels.count(0),
        labels.count(1),
        labels.count(2),
        #labels.count(3),
        #labels.count(4),
    ]
    plotTheClasses(graphic_bars)

    i = 0
    for l in labels:
        database["list"][i]["rating"] = l
        i+=1
    writeFile("./DataRetrieval/encoded_database.json",database)

def elbowmethod_2():
    from Tools.files import readFile
    database = readFile("./DataRetrieval/encoded_database.json")
    array_of_titles = getTitles(database)
    args = {
        "init": "random",
        "n_clusters": 3,
        "max_iter": 300,
        "random_state": 42}
    results = []
    for k in range(1,10):
        kmeans = KMeans(n_init= k, **args)
        kmeans.fit(array_of_titles)
        results.append(kmeans.inertia_)
    plotTheElbow(results,[1,2,3,4,5,6,7,8,9],"Elbow method")

    silhouette_coefficients = []
    for k in range(2,10):
        kmeans = KMeans(n_init= k, **args)
        kmeans.fit(array_of_titles)
        score = silhouette_score(array_of_titles, kmeans.labels_)
        silhouette_coefficients.append(score)
    plotTheElbow(silhouette_coefficients,[2,3,4,5,6,7,8,9],"shilouette coficient")

def elbowmethod():
    from Tools.files import readFile
    database = readFile("./DataRetrieval/encoded_database.json")
    array_of_titles = getTitles(database)
    args = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42}
    results = []
    for k in range(1,10):
        kmeans = KMeans(n_clusters= k, **args)
        kmeans.fit(array_of_titles)
        results.append(kmeans.inertia_)
    plotTheElbow(results,[1,2,3,4,5,6,7,8,9],"Elbow method")

    silhouette_coefficients = []
    for k in range(2,10):
        kmeans = KMeans(n_clusters= k, **args)
        kmeans.fit(array_of_titles)
        score = silhouette_score(array_of_titles, kmeans.labels_)
        silhouette_coefficients.append(score)
    plotTheElbow(silhouette_coefficients,[2,3,4,5,6,7,8,9],"shilouette coficient")
