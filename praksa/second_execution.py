import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def tokenize_and_lemmatize(text):
    # Remove stop words
    text = text.encode('ascii', 'ignore').decode('ascii') 
    words = nltk.tokenize.WhitespaceTokenizer().tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words).lower()

def text_classification(context, model_choice, tokenize_lemmatize=False):
    if tokenize_lemmatize:
        context = tokenize_and_lemmatize(context)

    if model_choice == 1:
        model_file = 'logisticregresion.pkl'
        vectorizer_file = 'vectorizer1.pkl'
    elif model_choice == 2:
        model_file = 'naivebayes.pkl'
        vectorizer_file = 'vectorizer2.pkl'
    elif model_choice == 3:
        model_file = 'kneighbour.pkl'
        vectorizer_file = 'vectorizer3.pkl'
    elif model_choice == 4:
        model_file = 'randomforest.pkl'
        vectorizer_file = 'vectorizer4.pkl'
    elif model_choice == 5:
        model_file = 'finalised_model.pkl'
        vectorizer_file = 'vectorizer5.pkl'
    else:
        print("Invalid model choice!")
        sys.exit(1)

    with open(model_file, 'rb') as file:
            model = pickle.load(file)

    with open(vectorizer_file, 'rb') as file:
            vectorizer = pickle.load(file)

    loaded_model = model
    loaded_vectorizer = vectorizer

    text_vectorized = loaded_vectorizer.transform([context])

    predicted_category = loaded_model.predict(text_vectorized)[0]
    print(f"Predicted category for '{context}': {predicted_category}")

    change_model = input("Do you want to use another model? (y/n): ").lower()
    if change_model == 'y':
        model_choice = int(input("Which machine learning model do you want to use?\n1) Logistic Regression\n2) Naive Bayes\n3) K-Nearest Neighbors\n4) Random Forest\n5) Support Vector Machine\n"))
        if model_choice not in {1, 2, 3, 4, 5}:
            print("Invalid choice!")
            sys.exit(1)
        text_classification(context, model_choice, tokenize_lemmatize=False)
    else:
        print('Goodbye!!!!')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    context = sys.argv[1]
    model_choice = int(input("Which machine learning model do you want to use?\n1) Logistic Regression\n2) Naive Bayes\n3) K-Nearest Neighbors\n4) Random Forest\n5) Support Vector Machine\n"))
    if model_choice not in {1, 2, 3, 4, 5}:
        print("Invalid choice!")
        sys.exit(1)
    text_classification(context, model_choice, tokenize_lemmatize=True)
