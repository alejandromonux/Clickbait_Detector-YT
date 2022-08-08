import os
import googleapiclient.discovery

from Tools.files import readFile, writeFile


def getVideoComments(name,id):
    # API information
    api_service_name = "youtube"
    api_version = "v3"
    # API key
    DEVELOPER_KEY = "AIzaSyAJ_jIg4eJM8hdWV9-6HXtB-60DoKzn4qc"
    # API client
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            maxResults=5,
            videoId=id
        )
        response = request.execute()
        for comment in response['items']:
            comments.append({"text": comment['snippet']['topLevelComment']['snippet']['textOriginal'],
                             "rating": 0})
    except Exception as e:
        print(e)
        print("problem with Video " + name + " / ID: " + id)
    return comments

def getChannelComments(name, id, database):
    # API information
    api_service_name = "youtube"
    api_version = "v3"
    # API key
    DEVELOPER_KEY = "AIzaSyAJ_jIg4eJM8hdWV9-6HXtB-60DoKzn4qc"
    # API client
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    # TODO: Diferenciar entre username e id (mirar guiones y tal)
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=id,
    )
    # Query execution
    responseChannel = request.execute()
    # Now with the response, we get the video list
    try:
        request = youtube.playlistItems().list(
            part="snippet",
            maxResults=50,
            playlistId=responseChannel['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        )
    except:
        print("problem with tuple " + name + "/" + id)
        return database

    response = request.execute()
    # Procesar el resultado y guardarlo en database
    index = 0
    for item in response['items']:
        for example in database["list"]:
            if item['snippet']['title'] == example["title"]:
                comments = getVideoComments(name=item['snippet']['title'],id=item['snippet']['resourceId']['videoId'])
                if len(comments)==0:
                    break
                else:
                    database["list"][database["list"].index(example)]["comments"] = comments
                    database["list"][database["list"].index(example)]["videoId"] = item['snippet']['resourceId']['videoId']
        index += 1

    return database

def getVideos(name, id, database):
    # API information
    api_service_name = "youtube"
    api_version = "v3"
    # API key
    DEVELOPER_KEY = "AIzaSyAJ_jIg4eJM8hdWV9-6HXtB-60DoKzn4qc"
    # API client
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    # TODO: Diferenciar entre username e id (mirar guiones y tal)
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=id,
    )
    # Query execution
    responseChannel = request.execute()
    # Now with the response, we get the video list
    try:
        request = youtube.playlistItems().list(
            part="snippet",
            maxResults=25,
            playlistId=responseChannel['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        )
    except:
        print("problem with tuple " + name + "/" + id)
        return database

    response = request.execute()
    # Procesar el resultado y guardarlo en database
    index = 0
    for item in response['items']:
        try:
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=item['snippet']['resourceId']['videoId']
            )
        except:
            print("problem with Video " + item['snippet']['title'] + "/ID: " + item['snippet']['resourceId'])
            return database
        infoVideo = request.execute()
        title = item['snippet']['title']
        description = item['snippet']['description']
        try:
            database['list'].insert(
                index,
                {
                    "author": responseChannel["items"][0]["snippet"]["title"],
                    "subscribers": responseChannel["items"][0]["statistics"]["subscriberCount"],
                    "title": title,
                    "description": description,
                    "category": infoVideo['items'][0]['snippet']['categoryId'],
                    "publishDate": infoVideo['items'][0]['snippet']['publishedAt'],
                    "views": infoVideo['items'][0]['statistics']['viewCount'],
                    "likes": infoVideo['items'][0]['statistics']['likeCount'],
                    "fav_count": infoVideo['items'][0]['statistics']['favoriteCount'],
                    "num_comments": infoVideo['items'][0]['statistics']['commentCount'],
                    "rating": "0"
                }
            )
        except Exception as e:
            print(e)
            print("problem with Video " + item['snippet']['title'] + " from " + responseChannel["items"][0]["snippet"][
                "title"] + " -No Statistics")
        # database["list"][index]["title"] = title
        # database["list"][index]["description"] = description
        # database["list"][index]["rating"] = 0
        index += 1

    return database

def commentsLoop():
    channels = readFile(os.getcwd() + "\channels.json")
    database = readFile(os.getcwd() + "/adjusted_database.json")

    for i in channels["channels"]:
        name = i["name"]
        id = i["id"]
        print(name + "\n")
        database = getChannelComments(name, id, database)

    writeFile(os.getcwd() + "/adjusted_database.json", database)
    print("I hem acabat!")

def channelLoop():
    """
    file = open("channels.json")
    channels = json.load(file)
    """
    channels = readFile(os.getcwd() + "\channels.json")
    """
    database = {
        "list":[
            {"title" : "a",
             "description": "a",
             "rating" : "1"}
        ]
    }
    """
    database = {}
    database["list"] = []

    for i in channels["channels"]:
        name = i["name"]
        id = i["id"]
        print(name + "\n")
        database = getVideos(name, id, database)

    # with open("database.json", "w") as file:
    #    json.dump(database,file)
    writeFile(os.getcwd() + "\database.json", database)

    print("I hem acabat!")

def dataForTheWeb(videoID):
    # API information
    api_service_name = "youtube"
    api_version = "v3"
    # API key
    DEVELOPER_KEY = "AIzaSyAJ_jIg4eJM8hdWV9-6HXtB-60DoKzn4qc"
    # API client
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    try:
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=videoID
        )
        infoVideo = request.execute()

        request = youtube.channels().list(
            part="snippet,contentDetails,statistics",
            id=infoVideo["items"][0]["snippet"]["channelId"],
        )
        # Query execution
        responseChannel = request.execute()

        video={
                    "author": responseChannel["items"][0]["snippet"]["title"],
                    "thumbnail":  infoVideo['items'][0]['snippet']["thumbnails"]["medium"]["url"],
                    "subscribers": responseChannel["items"][0]["statistics"]["subscriberCount"],
                    "title": infoVideo['items'][0]['snippet']["title"],
                    "description": infoVideo['items'][0]['snippet']["description"],
                    "category": infoVideo['items'][0]['snippet']['categoryId'],
                    "publishDate": infoVideo['items'][0]['snippet']['publishedAt'],
                    "views": infoVideo['items'][0]['statistics']['viewCount'],
                    "likes": infoVideo['items'][0]['statistics']['likeCount'],
                    "fav_count": infoVideo['items'][0]['statistics']['favoriteCount'],
                    "num_comments": infoVideo['items'][0]['statistics']['commentCount'],
                    "rating": "0"
                }
    except Exception as e:
        print("problem with Video " + videoID)
        return 0
    video["comments"]=getVideoComments(video["author"],videoID)
    return video