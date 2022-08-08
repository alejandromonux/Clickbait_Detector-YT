from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class videoForm(FlaskForm):
    videoId = StringField('videoId', validators=[DataRequired()])
    submit = SubmitField('Analiza el video')