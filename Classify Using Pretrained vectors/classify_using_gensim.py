import numpy as np
import gensim
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = 200

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    

w2v = gensim.models.Word2Vec.load('<path_to_gensim_model>')

w2v_classify = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])

#Create a list of Fruits and vegetables
X = [['apple', 'mango', 'papaya'],
     ['potato', 'tomato', 'onion']]

#labels
y = ['fruits', 'vegetables']

#fit the data
w2v_classify.fit(X, y)

#test with new text
test_X = [['banana'], ['cabbage']]
print (w2v_classify.predict(test_X))
