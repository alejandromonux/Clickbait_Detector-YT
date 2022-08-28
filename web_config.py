import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'e8860ae3191c777768973ed96a204bf6e06e7cb696259e5ae2674d4028d41c2b'
    FLASK_ENV = "production"