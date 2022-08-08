import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_encoding(titles):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(titles)
    matrix_pd = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())
    outArray = []
    for i in range(matrix_pd.shape[0]):
        outArray.append(matrix_pd.iloc[i].values)
    return outArray