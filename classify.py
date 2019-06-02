#!/bin/python

def train_classifier(X, y, C=1.0):
    """
    Train a classifier using the given training data.

    Trains logistic regression on the input data with default parameters.
    """
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(C = C, random_state=0, solver='lbfgs', max_iter=10000)
    cls.fit(X, y)
    return cls

def evaluate(X, yt, cls, name='data', verbose = True):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics
    yp = cls.predict(X)
    predict_prob = cls.predict_proba(X)
    for i in range(len(yt)):
            if yp[i] != yt[i] and predict_prob[i][0]>0.90:
                print(i+1)
#     print(predict_prob)
    acc = metrics.accuracy_score(yt, yp)
    if verbose:
        print("  Accuracy on %s is: %s" % (name, acc))
        
    return acc
    