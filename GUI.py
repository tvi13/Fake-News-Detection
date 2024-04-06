import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

st.title("Fake News Detection")

# Load the trained models
LR = LogisticRegression()
MB =MultinomialNB()
GBC = GradientBoostingClassifier(random_state=0)


# Load TF-IDF vectorizer
vectorization = TfidfVectorizer()

# Load the manual testing data
df_manual_testing = pd.read_csv("/Users/tvishamajithia/Desktop/manual_testing.csv")

# Fit TF-IDF vectorizer
x = df_manual_testing["text"]
y = df_manual_testing["class"]
xv = vectorization.fit_transform(x)

# Fit the models
LR.fit(xv, y)
MB.fit(xv, y)
GBC.fit(xv, y)


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def predict_news(news):
    news = wordopt(news)
    new_x_test = [news]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_MB = MB.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    
    return pred_LR[0], pred_MB[0], pred_GBC[0]

# User input text
user_input = st.text_area("Enter the news text:")

if st.button("Predict"):
    if user_input:
        lr, mb, gbc = predict_news(user_input)
        st.write("Logistic Regression Prediction:", output_lable(lr))
        st.write("Naive Bayes Classifier:", output_lable(mb))
        st.write("Gradient Boost Prediction:", output_lable(gbc))
        
    else:
        st.error("Please enter some news text.")
