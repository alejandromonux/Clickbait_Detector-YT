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
    #Obtenemos el formulario
    form = videoForm()
    error = request.args.get('error', None)
    #comprovamos si el formulario ha sido rellenado o no
    if form.validate_on_submit():
        query=form.videoId.data
        #Pasamos a la pantalla de resultados
        return redirect(url_for('results', url=query))
    from flask import make_response
    #Seteamos cookies si hace falta
    if request.cookies.get('session') == None:
        response = make_response(render_template('index.html', title='Home', form=form))
        from app.__init__ import userCounter
        userCounter += 1
        response.set_cookie(key='session',value=str(userCounter))
    else:
        #Obtenemos la lista de videos para la pantalla de home
        try:
            videolist = cache[request.cookies.get('session')]
            response = make_response(render_template('index.html', title='Home', form=form,lastRevisions=videolist,error=error))
        except:
            response = make_response(render_template('index.html', title='Home', form=form, error=error))
    return response

@app.route('/results')
def results():
    name = request.cookies.get('session')
    url = request.args.get('url', None)
    base_url = request.url
    #We check if the user gave feedback and where
    from flask import redirect
    if "submit" in base_url:
        opinion=base_url.split("submit=I+")[1]
        if opinion=="agree":
            print("Agreed")
            user_opinion =cache[name][len(cache[name]) - 1]["rating"]
        elif opinion=="disagree":
            print("Disagreed")
            user_opinion= 1-cache[name][len(cache[name])-1]["rating"]

        cache[name][len(cache[name])-1]["opinion"] = user_opinion
        from Tools.logs import modifyLogs
        modifyLogs(os.getcwd()+"\\DataRetrieval\\webRequestLogs.json",cache[name][len(cache[name])-1], user_opinion)
        return redirect(url_for('index'))

    #We analize the video
    url=getURL(url)
    video = dataForTheWeb(url)
    if video == 0:
        print("ERROR CON EL VIDEO")
        return redirect(url_for('index', error="This video cannot be processed. Please try another"))
    from Tools.logs import appendToLogs

    from Tools.Preprocessing import arrayBERTPreprocessing
    predictionObject = arrayBERTPreprocessing([video],[0])

    pred,y_proba,y_feat_proba,neg,neu,pos = model.predict(predictionObject["x"][0])
    video["opinion"]=pred
    appendToLogs(os.getcwd()+"\\DataRetrieval\\webRequestLogs.json",video, pred)
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
    from app.forms import FeedbackFormPositive
    formPos= FeedbackFormPositive()
    from app.forms import FeedbackFormNegative
    formNeg=FeedbackFormNegative()
    return render_template('results.html', title='Result', result=data, formpos=formPos,formneg=formNeg)

@app.route('/about')
def about():
    return render_template('about.html', title='About')