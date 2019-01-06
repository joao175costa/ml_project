import numpy as np
from readers import read_dataset
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def classic_classifiers():
    X, Y = read_dataset()

    # quick test with SVM
    classifier = SVC()
    classifier.fit(X, Y)
    print(classification_report(Y, classifier.predict(X)))