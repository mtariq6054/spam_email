import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Custom CSS for button hover color and page background
custom_css = """
<style>
.stButton>button:hover {
    background-color: #2980b9 !important;
}
.stTextInput>div>div>textarea {
    background-color: rgba(255, 255, 255, 0.7) !important;
    color: black !important;
    border-radius: 10px;
    padding: 20px;
    border: 4px solid #2980b9 !important;
}
.stTextInput>div>div>textarea::placeholder {
    color: gray;
    font-style: italic;
}
.prediction-result {
    padding: 15px;
    border: 3px solid #2980b9;
    border-radius: 10px;
    background-color: #ecf0f1;
    font-weight: bold;
    font-family : 'Time New Roman';
}
.prediction-button {
    font-weight: bold !important;
    font-family:  'Time New Roman' ;
    font-size: 40px;
    font-weight: bold;
}
.prediction-probability {
    margin-top: 10px;
    font-size: 20px;
    font-weight: bold;
    color: #2980b9;
    border: 2px solid #2980b9;
    border-radius: 5px;
    padding: 5px 10px;
    background-color: #ecf0f1;
}
.title {
    font-family:  'Time New Roman' ;
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #074173;
}
.subheader {
    font-family: 'Time New Roman' ;
    font-size: 27px;
    font-weight: bold;
    color: #074173;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.wallpapertip.com/wmimgs/35-358745_thumb-image-email-background-image-hd.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the Naive Bayes model
naive_bayes_model = joblib.load('Naive Bayes model.pkl')

# Load the CountVectorizer
count_vectorizer = joblib.load('count_vectorizer.pkl')

# Define a function to predict spam probability
def predict_spam(email_text):
    # If CountVectorizer is used, transform the text
    if count_vectorizer:
        email_features = count_vectorizer.transform([email_text])
    else:
        email_features = [email_text]
    # Predict spam probability
    spam_probability = naive_bayes_model.predict_proba(email_features)[0][1]
    return spam_probability

# Streamlit app
st.markdown('<h1 class="title">Spam Email Classifier</h1>', unsafe_allow_html=True)

# Subheader for user to input email text
st.markdown('<h2 class="subheader">Enter your email text here:</h2>', unsafe_allow_html=True)
email_text = st.text_area("", placeholder="Type your email here...")

# Button to trigger prediction
if st.button("Predict", key="predict_button"):
    spam_probability = predict_spam(email_text)
    st.write("Spam Probability:", spam_probability)

    # Present the prediction result
    st.subheader("Prediction Result:")
    result_text = "This email is likely spam." if spam_probability > 0.7 else "This email is likely not spam."
    st.markdown(f'<div class="prediction-result">{result_text}</div>', unsafe_allow_html=True)
