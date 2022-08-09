import os
from urllib import request

from flask import render_template, url_for, request
from app.__init__ import model, cache
from DataRetrieval.yt_api import dataForTheWeb, getURL
from app import app

@app.route('/',  methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    from flask import render_template, flash, redirect
    from app.forms import videoForm
    form = videoForm()
    if form.validate_on_submit():
        query=form.videoId.data
        return redirect(url_for('results', url=query))
    from flask import make_response
    if request.cookies.get('session') == None:
        response = make_response(render_template('index.html', title='Home', form=form))
        from app.__init__ import userCounter
        userCounter += 1
        response.set_cookie(key='session',value=str(userCounter))
    else:
        try:
            videolist = cache[request.cookies.get('session')]
            response = make_response(render_template('index.html', title='Home', form=form,lastRevisions=videolist))
        except:
            response = make_response(render_template('index.html', title='Home', form=form))
    return response

@app.route('/results')
def results():
    name = request.cookies.get('session')

    url = request.args.get('url', None)
    url=getURL(url)
    video = dataForTheWeb(url)
    from Tools.logs import appendToLogs
    appendToLogs(os.getcwd()+"\\DataRetrieval\\webRequestLogs.json",video)
    from Tools.Preprocessing import arrayBERTPreprocessing
    predictionObject = arrayBERTPreprocessing([video],[0])

    pred,y_proba,y_feat_proba,neg,neu,pos = model.predict(predictionObject["x"][0])
    try:
        feat_prob=0 if y_feat_proba[0][0] == 0 and y_feat_proba[0][1] == 0 else 1
        feat_proba_0 = y_feat_proba[0][0]
        feat_proba_1 = y_feat_proba[0][1]
    except:
        feat_prob = 0 if y_feat_proba[0] == 0 and y_feat_proba[1] == 0 else 1
        feat_proba_0 = y_feat_proba[0]
        feat_proba_1 = y_feat_proba[1]
    sentiment=0 if neg == 0 and neu == 0 and pos==0 else 1

    data = {
        "thumbnail":     video["thumbnail"],
        "title":         video["title"],
        "rating":        pred,
        "title_prob_0":  y_proba[0][0],
        "title_prob_1":  y_proba[0][1],
        "feat_prob":     feat_prob,
        "feat_prob_0":   feat_proba_0,
        "feat_prob_1":   feat_proba_1,
        "sentiment":     sentiment,
        "sentiment_neg": neg,
        "sentiment_neu": neu,
        "sentiment_pos": pos
    }
    try:
        n_cache_item={
            "thumbnail": video["thumbnail"],
            "title":     video["title"],
            "rating":    pred,
            "author":    video["author"],
            "views":     video["views"],
            "likes":     video["likes"]
        }
        cache[name].append(n_cache_item)
    except:
        cache[name]=[]
        cache[name].append(n_cache_item)

    return render_template('results.html', title='Result', result=data)

@app.route('/about')
def about():
    return render_template('about.html', title='About')