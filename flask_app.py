from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd
import numpy as np


 # Rename your Flask application
flask_app = Flask(__name__, template_folder='template')

# flask_app = Flask(__name__) 

# Load the Naive Bayes model
naive_bayes_model = joblib.load('Naive Bayes model.pkl')

# Load the CountVectorizer
count_vectorizer = joblib.load('count_vectorizer.pkl')

# Define function to predict spam probability 
def predict_spam(email_text):
    # If CountVectorizer is used, transform the text
    if count_vectorizer:
        email_features = count_vectorizer.transform([email_text])
    else:
        email_features = [email_text]

    # Predict spam probability 
    spam_probability = naive_bayes_model.predict_proba(email_features)[0][1]
    return spam_probability

# Create routes
@flask_app.route('/')
def index():
    return render_template('anim.html')


@flask_app.route('/predict', methods=['POST'])
def predict():
    return render_template('index.html' )


# @flask_app.route('/predict', methods=['POST'])
# def predict():
#     email_text = request.form['email_text']
#     spam_probability = predict_spam(email_text)
#     return render_template('template/index.html', spam_probability=spam_probability)



# @flask_app.route('/classify', methods=['POST'])
# def classify():
#     # email_text = request.form['email_text']
#     # spam_probability = predict_spam(email_text)
#     return render_template('index.html',spam_probability = spam_probability )


if __name__ == '__main__':
    flask_app.run(debug=True)
