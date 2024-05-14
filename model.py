import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib 

# load the dataset
email_df = pd.read_csv(r'C:\Users\12\Documents\Email Classification Project\spam.csv', quotechar='"', delimiter=',', encoding='latin1', skip_blank_lines=True)

# change the name of column
new_columns_name = {'v1':'label', 'v2':'text'}
email_df.rename(columns=new_columns_name, inplace=True)

email_df.isnull().sum()

email_df.drop("Unnamed: 2", axis=1 , inplace =True)

email_df.drop("Unnamed: 3", axis=1 , inplace =True)
email_df.drop("Unnamed: 4", axis=1 , inplace =True)

# Start pre-processing on data set 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(email_df['text'])

# get labels 
y = email_df['label']

# spliting for modeling 
X_train, X_test , y_train, y_test = train_test_split(X, y , test_size= 0.10 , random_state = 42)


# Training of Multinomial naive bayes model 
Nb = MultinomialNB()
Nb.fit(X_train, y_train)

# predictions on testing data 
Nb_pred = Nb.predict(X_test)

# evaluation of model
Nb_pred_accu = accuracy_score(y_test, Nb_pred)
print("Accuracy is: ",Nb_pred_accu )

# *************Confusion Matrix**********************
Nb_pred_conf = confusion_matrix(y_test,Nb_pred )
print("Confusion matrix score is: ",Nb_pred_conf )



# ************ Save the model for future use
joblib.dump(Nb ,"Naive Bayes model.pkl")

joblib.dump(vectorizer , 'count_vectorizer.pkl')