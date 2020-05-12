import logging
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class MachineLearningAlgorithm:

    def vectorizer(self,df,text_column,target_column):
        tfidf = TfidfVectorizer(stop_words='english')
        X = tfidf.fit_transform(df[text_column].values.astype('U'))
        X_train, X_test, y_train, y_test = train_test_split(X, df[target_column], test_size = 0.33, random_state = 42)
        return tfidf, X_train, X_test, y_train, y_test

    def LogisticRegressionModel(self,X_train, X_test, y_train, y_test):
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        logging.info('Test Score %s',lr.score(X_test, y_test))
        return lr, lr.score(X_test, y_test)

    def RandomForestModel(self,X_train, X_test, y_train, y_test):
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        logging.info('Test Score %s', rf.score(X_test, y_test))
        return rf, rf.score(X_test, y_test)


    def NaiveBayesModel(self,X_train, X_test, y_train, y_test):
        nvb = MultinomialNB()
        nvb.fit(X_train, y_train)
        logging.info('Test Score %s', nvb.score(X_test, y_test))
        return nvb, nvb.score(X_test, y_test)


    def LinearSVCModel(self,X_train, X_test, y_train, y_test):
        lsv = LinearSVC()
        lsv.fit(X_train, y_train)
        logging.info('Test Score %s', lsv.score(X_test, y_test))
        return lsv, lsv.score(X_test, y_test)


    def DecisionTreeModel(self,X_train, X_test, y_train, y_test):
        dec = DecisionTreeClassifier()
        dec.fit(X_train, y_train)
        logging.info('Test Score %s', dec.score(X_test, y_test))
        return dec, dec.score(X_test, y_test)


    def XGBoostModel(self,X_train, X_test, y_train, y_test):
        xg = XGBClassifier()
        xg.fit(X_train, y_train)
        logging.info('Test Score %s', xg.score(X_test, y_test))
        return xg, xg.score(X_test, y_test)


    def SGDModel(self,X_train, X_test, y_train, y_test):
        sgdd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
        sgdd.fit(X_train, y_train)
        logging.info('Test Score %s', sgdd.score(X_test, y_test))
        return sgdd, sgdd.score(X_test, y_test)

    def predict_target(self,vectorizer, text, model):
        data = [text]
        vect = vectorizer.transform(data).toarray()
        logging.info('Predicted class %s', model.predict(vect))
        return model.predict(vect)
