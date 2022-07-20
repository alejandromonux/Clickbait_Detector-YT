import math
import os

import matplotlib
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from Tools.files import writeFile, readFile


def plot3D(centroids, data, labels,classes):
    colors = ['brown', 'purple', 'cyan', 'green', 'orange']
    colors_centroids = ['b', 'g', 'r', 'm', 'y']

    import pandas as pd
    import numpy as np
    import plotly.express as px
    df = pd.DataFrame({'cat': labels, 'col_x': data[:,0], 'col_y': data[:,1], 'col_z': data[:,2]})
    fig = px.scatter_3d(df,
                        x='col_x', y='col_y', z='col_z',
                        color='cat',
                        title="Classes 3D")
    fig.show()


def resultsAnalysis(model, data, labels, classes):
    colors = ['brown', 'purple', 'cyan', 'green', 'orange']
    colors_centroids = ['b', 'g', 'r', 'm', 'y']
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    data_transformed = model.fit_transform(data)
    centroids = model.cluster_centers_

    for i in range(classes):
        if classes == 3:
            plot3D(centroids,data_transformed,labels,classes)
            return
        import numpy as np
        group_x = []
        group_y = []
        for j in range(len(data_transformed)):
            if labels[j] == i:
                group_x.append(data_transformed[j][0])
                group_y.append(data_transformed[j][1])
        plt.scatter(group_x, group_y, color=colors[i])
        this_centroid = [np.average(group_x)+centroids[i][0],np.average(group_y)+centroids[i][1]]
        plt.scatter(this_centroid[0], this_centroid[1], marker="X", s=120, color=colors_centroids[i])
        plt.show()

        #Revisar los ejemplos que se salen de la media y analizarlos
        print("CLASE %s" ,(i))
        titulos = [[],[]]
        database = readFile(os.getcwd() + "\database.json")
        for item in database["list"]:
            if item["rating"] == i:
                titulos[i].append(item["title"])

        acum_dist = []
        for j in range(len(group_x)):
            point = [group_x[j], group_y[j]]
            from scipy.spatial import distance
            acum_dist.append(distance.euclidean(point,this_centroid))
        from numpy import average
        avg = average(acum_dist)
        for i in range(len(acum_dist)):
            if acum_dist[i] > avg:
                print(titulos[i][:])
    print("DEBUG")


def getTitles(database):
    out = []
    i = 0
    for item in database["list"]:
        out.insert(i, item["title"])
        i += 1
    return out


def plotTheClasses(graphic_bars):
    figure = plt.figure()
    ax = figure.add_axes([0, 0, 1, 1])
    values = []
    for i in range(0, len(graphic_bars)):
        values.append(str(i))  # = ['0', '1', '2', '3','4']
    ax.bar(values, graphic_bars)
    plt.show()


def plotTheElbow(y_values, x_value, title):
    plt.plot(x_value, y_values)
    plt.xlabel("Valor para K")
    plt.ylabel("SSE")
    plt.title(title)
    plt.show()


def kmeans_test(k_clusters):
    from Tools.files import readFile
    database = readFile(os.getcwd() + "\encoded_database.json")
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
    graphic_bars = []
    for i in range(0, k_clusters):
        graphic_bars.append(labels.count(i))
    plotTheClasses(graphic_bars)
    i = 0
    resultsAnalysis(kmeans, array_of_titles, labels, k_clusters)
    for l in labels:
        database["list"][i]["rating"] = l
        i += 1
    writeFile(os.getcwd() + "\encoded_database.json", database)


def elbowmethod_2():
    from Tools.files import readFile
    database = readFile(os.getcwd() + "\encoded_database.json")
    array_of_titles = getTitles(database)
    args = {
        "init": "random",
        "n_clusters": 3,
        "max_iter": 300,
        "random_state": 42}
    results = []
    for k in range(1, 10):
        kmeans = KMeans(n_init=k, **args)
        kmeans.fit(array_of_titles)
        results.append(kmeans.inertia_)
    plotTheElbow(results, [1, 2, 3, 4, 5, 6, 7, 8, 9], "Elbow method")

    silhouette_coefficients = []
    for k in range(2, 10):
        kmeans = KMeans(n_init=k, **args)
        kmeans.fit(array_of_titles)
        score = silhouette_score(array_of_titles, kmeans.labels_)
        silhouette_coefficients.append(score)
    plotTheElbow(silhouette_coefficients, [2, 3, 4, 5, 6, 7, 8, 9], "shilouette coficient")


def elbowmethod():
    from Tools.files import readFile
    database = readFile(os.getcwd() + "\encoded_database.json")
    array_of_titles = getTitles(database)
    args = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42}
    results = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, **args)
        kmeans.fit(array_of_titles)
        results.append(kmeans.inertia_)
    plotTheElbow(results, [1, 2, 3, 4, 5, 6, 7, 8, 9], "Elbow method")

    silhouette_coefficients = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, **args)
        kmeans.fit(array_of_titles)
        score = silhouette_score(array_of_titles, kmeans.labels_)
        silhouette_coefficients.append(score)
    plotTheElbow(silhouette_coefficients, [2, 3, 4, 5, 6, 7, 8, 9], "shilouette coficient")
