import numpy as np;
import pandas as pd;
import csv
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import nltk
from nltk.stem import 	WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

file = "bbc-news-data.csv"
data = pd.read_csv(file, sep="\t")

def tokenize_and_lemmatize(text):
    data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    words = nltk.tokenize.WhitespaceTokenizer().tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words).lower()

data['new content'] = data['content'].apply(tokenize_and_lemmatize)

X = data["new content"]
y = data["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

vectorizer_count = CountVectorizer()
X_train_vectorized = vectorizer_count.fit_transform(X_train)
X_test_vectorized = vectorizer_count.transform(X_test)

model_nb = MultinomialNB()
model_nb.fit(X_train_vectorized, y_train)

model_knn = KNeighborsClassifier()
model_knn.fit(X_train_features, y_train)

model_svc = SVC()
model_svc.fit(X_train_features, y_train)

model_rf = RandomForestClassifier()
model_rf.fit(X_train_features, y_train)

model_lr = LogisticRegression()
model_lr.fit(X_train_vectorized, y_train)

models = [model_nb, model_knn, model_svc, model_rf, model_lr]
filenames = ['naivebayes.sav', 'kneighbour.pkl', 'finalised_model.pkl', 'randomforest.pkl', 'logisticregresion.pkl']
vectorizers = [vectorizer_count, vectorizer, vectorizer, vectorizer, vectorizer_count]
vectorizer_files = ['vectorizer.pkl', 'vectorised.pkl', 'svcvectorised.pkl', 'rfvectorizer.pkl', 'vectorizer.pkl']

for model, filename, vectorizer, vectorizer_file in zip(models, filenames, vectorizers, vectorizer_files):
    pickle.dump(model, open(filename, 'wb'))
    pickle.dump(vectorizer, open(vectorizer_file, 'wb'))

context = input("Input the text you want to categorise: ")

def question():
    print("Which machine learning model do you want to use:")
    print("1) Logistic Regression")
    print("2) Naive Bayes ")
    print("3) K-Nearest Neighbors")
    print("4) Random Forest")
    print("5) Support Vector Machine")

question()

ml = input("Choose which model you want to use: ")
if ml not in {'1', '2', '3', '4', '5'}:
    print("Invalid choice!")
    question() 
else:
    if ml == '1':
        filename = 'logisticregresion.pkl'
        filename_vectorizer = 'vectoriser.pkl'
    elif ml == '2':
        filename = 'naivebayes.sav'
        filename_vectorizer = 'vectorizer.pkl'
    elif ml == '3':
        filename = 'kneighbour.pkl'
        filename_vectorizer = 'svcvectorised.pkl'
    elif ml == '4':
        filename = 'randomforest.pkl'
        filename_vectorizer = 'rfvectorizer.pkl'
    elif ml == '5':
        filename = 'finalised_model.pkl'
        filename_vectorizer = 'vectorised.pkl'

loaded_model = pickle.load(open(filename, 'rb'))
loaded_vectorizer = pickle.load(open(filename_vectorizer, 'rb'))

text_vectorized = loaded_vectorizer.transform([context])
predicted_category = loaded_model.predict(text_vectorized)[0]
print(f"Predicted category for '{context}': {predicted_category}")
