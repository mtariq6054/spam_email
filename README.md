Headings: Use # for headings (# Heading 1, ## Heading 2, etc.).

Text formatting: You can make text bold (**bold text**), italic (*italic text*), or add inline code ( inline code ).

Lists: Create ordered lists (1. Item 1, 2. Item 2) or unordered lists (- Item 1, - Item 2).

Code blocks: Use triple backticks (```) to create code blocks.

Links: [Link text](URL) to create clickable links.

Images: ![Alt text](image URL) to embed images.


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

