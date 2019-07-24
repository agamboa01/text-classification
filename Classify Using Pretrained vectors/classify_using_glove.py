import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
with open("glove.6B.50d.txt", "rb") as lines:
    glove_w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}
    
etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(glove_w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])

#Create a list of Fruits and vegetables
X = [['apple', 'mango', 'papaya'],
     ['potato', 'tomato', 'onion']]

#labels
y = ['fruits', 'vegetables']

#fit the data
etree_w2v.fit(X, y)

#test with new text
test_X = [['banana'], ['cabbage']]
print (etree_w2v.predict(test_X))
