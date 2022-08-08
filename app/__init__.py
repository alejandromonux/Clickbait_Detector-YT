import os

from flask import Flask
from flask_bootstrap import Bootstrap5

from Classification.Model.FinalModel import Model
from Tools.files import readFile
from web_config import Config
"""---VARIABLES---"""
userCounter = 0
cache = {}
file_route="\\DataRetrieval\\adjusted_database.json"
model = Model(readFile("\\DataRetrieval\\adjusted_database.json"), [[],[]],willImport=True)
model.loadModel(prefix="\\DataRetrieval\\")

"""---FLASK PROCESSES---"""
app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap5(app)

from app import routes
