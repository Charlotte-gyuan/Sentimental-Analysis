

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.base import TransformerMixin

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.toarray()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


if __name__ == '__main__':


    with open("training_dataset.txt", 'r') as f:
        file = f.read()
    lines = file.split('\n')
    y_train = []
    documents_train= []
    for line in lines:
        documents_train.append(line[2:])
        y_train.append(line[0])


    # shufle training data
    indices = np.random.permutation(len(documents_train))
    documents_train = np.array(documents_train)[indices]
    y_train = np.array(y_train)[indices]


    cv = 1
    classifier_pipeline = make_pipeline(CountVectorizer(ngram_range=(2, 3)),
                                        DenseTransformer(),
                                        LinearSVC())

    if (cv == 1):
        # If use cross validation
        scores = cross_val_score(classifier_pipeline, documents_train, y_train, cv=5, scoring='f1_weighted')
        print("Average f1_weighted scores: ", scores.mean())
    else:
        # Train the model by ourselves and predict
        clf = classifier_pipeline.fit(documents_train[:-10], y_train[:-10])  # Choose our training data
        testdata = classifier_pipeline.named_steps.CountVectorizer.transform(documents_train[-2:]).toarray()
        print("testing data vector:\n", testdata, "\nprediction:\n",
              classifier_pipeline.named_steps.LinearSVC.predict(testdata))
