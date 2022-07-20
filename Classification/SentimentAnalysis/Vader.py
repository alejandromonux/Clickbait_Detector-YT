import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Tools.Preprocessing import textCleanupForVader


def sentimentAnalysis(database):
    analyzer = SentimentIntensityAnalyzer()

    for j in range(len(database["list"])):
        try:
            for i in range(len(database["list"][j]["comments"])):
                #item["comments"][i]["text"] = textCleanupForVader(item["comments"][i]["text"])
                database["list"][j]["comments"][i]["rating"] = analyzeOnePhrase(database["list"][j]["comments"][i]["text"])
                print(database["list"][j]["comments"][i]["text"] + " is: "+ str(database["list"][j]["comments"][i]["rating"]))
        except Exception as e:
            print(e)


def analyzeOnePhrase(text):
    analyzer = SentimentIntensityAnalyzer()
    text = textCleanupForVader(text)
    return analyzer.polarity_scores(text)