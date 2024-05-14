
# Spam Email Classifier Documentation

**This documentation provides an overview of the Spam Email Classifier project, which aims to classify emails as spam or not spam using machine learning techniques. The project includes building a classification model and integrating this model into a web application using streamlit , and creating a user interface for email classification.**

# Project Overview
1. Data Collection 
2. Perform EDA on data
3. Perform preprocessing 
4. Choose Machine Learning algorithm 
5. After choosing train model on preprocess data
6. Get predictions on testing data
7. Perform Evaluation metrics on predictions
8. Choose web frame 
9. Deploment and testing
 
# Start building Project
# Data Collection and Preprocessing
# Dataset:
**The project utilizes a dataset containing email texts labeled as spam or not spam.**
# Preprocessing Steps:
- Loading the dataset
- Perform EDA for taking overview of data and also check and remove irrelevent things
- Preprocessing of data as in we handle text data , in that case we apply counter vectorizer technique or any other technique 
- Splitting the data


# Model Building and Training
# Algorithm: Multinomial Naive Bayes
**Training Process:**
- Splitting the dataset into training and testing sets
- Training the model on the training data
- Evaluating the model's performance on the testing data

# Perform different Evaluation metrics 
**On classification model**
- Accuracy 
- Precision
- Recall
- f1-Score
- Confusion Matrix

# Streamlit Web Framework 
- To avoid external html , css
- for quick results 
- using small functions

 Links: [Learn about Count Vectorizer]( https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) .


 Images: ![Flow of Count Vectorizer](https://www.researchgate.net/publication/354354484/figure/fig2/AS:1080214163595268@1634554534648/Illustration-of-count-vectorization.jpg) .

  Images: ![Flow of Count Vectorizer](https://www.educative.io/api/edpresso/shot/5197621598617600/image/6596233398321152) .