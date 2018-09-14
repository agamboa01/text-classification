# - b : business - t : science and technology - e : entertainment - m : health
"""
--pre-requisites:
nltk
pandas
sklearn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
stemmer = PorterStemmer()

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

def train(classifier, X, y,predict_text):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=33)
    classifier.fit(X_train, y_train)
    prediction = (classifier.predict(predict_text))
    print ("Accuracy: %s" % classifier.score(X_test, y_test))
    print ("predicted category: ",prediction)

df = pd.read_csv("/home/ekbana/workspace/LOCAL/text-classification-using-machine-learning-algorithms/train_data.csv")
predict_text = ["How 'fast money' crushed 'Candy Crush' IPO"]
actual_category = "b"

print ("=========NAIVE MODEL============")
naive_model = Pipeline([('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer, stop_words=stopwords.words('english') + list(string.punctuation))),('classifier', MultinomialNB(alpha=0.05))])
print ("actual category", actual_category)
train(naive_model, df['TITLE'], df['CATEGORY'],predict_text)

print ("=========LogisticRegression MODEL============")
logistic_model = Pipeline([('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer, stop_words=stopwords.words('english') + list(string.punctuation))),('classifier', LogisticRegression())])
print ("actual category", actual_category)
train(logistic_model, df['TITLE'], df['CATEGORY'],predict_text)


"""
OUTPUT:
=========NAIVE MODEL============
actual category b
Accuracy: 0.9211567962727979
predicted category:  ['b']
=========LogisticRegression MODEL============
actual category b
Accuracy: 0.9251624022272306
predicted category:  ['b']
"""
