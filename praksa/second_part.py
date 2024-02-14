import numpy as np
import pandas as pd
import csv
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import sys

def tokenize_and_lemmatize(text):
    text = text.encode('ascii', 'ignore').decode('ascii') 
    words = nltk.tokenize.WhitespaceTokenizer().tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words).lower()

def text_classification(context, **kwargs):
    tokenize_lemmatize = kwargs.get('tokenize_lemmatize', False)
    if tokenize_lemmatize:
        context = tokenize_and_lemmatize(context)

    file = "bbc-news-data.csv"
    data = pd.read_csv(file, sep="\t")

    data['new content'] = data['content'].apply(tokenize_and_lemmatize)

    X = data["new content"]
    y = data["category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    vectorizer_file = 'vectorizer.pkl'

    vectorizer = TfidfVectorizer()
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    with open(vectorizer_file, 'wb') as file:
        pickle.dump(vectorizer, file)


    model_choice = kwargs.get('model_choice')

    if model_choice == 'Logistic Regression':
        model = LogisticRegression()
    elif model_choice == 'Naïve Bayes':
        model = MultinomialNB()
    elif model_choice == 'K-Nearest Neighbors':
        model = KNeighborsClassifier()
    elif model_choice == 'Random Forest':
        model = RandomForestClassifier()
    elif model_choice == 'Support Vector Machine':
        model = SVC()

    model.fit(X_train_features, y_train)
    
    pickle.dump(model, open('chosen_model.pkl', 'wb'))

    loaded_model = pickle.load(open('chosen_model.pkl', 'rb'))
    loaded_vectorizer = pickle.load(open(vectorizer_file, 'rb'))

    text_vectorized = loaded_vectorizer.transform([context])
    predicted_category = loaded_model.predict(text_vectorized)[0]
    print(f"Predicted category for '{context}': {predicted_category}")

    change_model = input("Do you want to use another model? (y/n): ").lower()
    if change_model == 'y':
        model_choice = input("Which machine learning model do you want to use?\n1) Logistic Regression\n2) Naïve Bayes\n3) K-Nearest Neighbors\n4) Random Forest\n5) Support Vector Machine\n")
        if model_choice not in {'1', '2', '3', '4', '5'}:
            print("Invalid choice!")
            sys.exit(1)
        models = ['Logistic Regression', 'Naïve Bayes', 'K-Nearest Neighbors', 'Random Forest', 'Support Vector Machine']
        text_classification(context, model_choice=models[int(model_choice)-1], tokenize_lemmatize=False)
    else:
        print('Goodbye!!!!')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    context = sys.argv[1]
    model_choice = input("Which machine learning model do you want to use?\n1) Logistic Regression\n2) Naïve Bayes\n3) K-Nearest Neighbors\n4) Random Forest\n5) Support Vector Machine\n")
    if model_choice not in {'1', '2', '3', '4', '5'}:
        print("Invalid choice!")
        sys.exit(1)
    models = ['Logistic Regression', 'Naïve Bayes', 'K-Nearest Neighbors', 'Random Forest', 'Support Vector Machine']
    text_classification(context, model_choice=models[int(model_choice)-1], tokenize_lemmatize=True)
