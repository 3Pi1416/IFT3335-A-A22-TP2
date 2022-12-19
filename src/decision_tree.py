import os
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def decision_tree(x_train: list, x_test: list, y_train: list, y_test: list, ngram: int = 1, use_stoplist: bool = False) -> int:
    
    if (use_stoplist):
        stoplist_file = Path(os.getcwd()).joinpath('stoplist-english.txt')
        with open(stoplist_file, mode='r') as f:
            stoplist_string = f.read()
        stoplist = stoplist_string.split("\n")
        model = make_pipeline(TfidfVectorizer(stop_words=stoplist, ngram_range=(ngram, ngram)), DecisionTreeClassifier())
    else: model = make_pipeline(TfidfVectorizer(ngram_range=(ngram, ngram)), DecisionTreeClassifier())

    model.fit(x_train, y_train)

    y_pred = np.array(model.predict(x_test))
    y_test = np.array(y_test)

    mispredicted = sum(y_test != y_pred)

    print("Decision tree")
    print("Number of mispredicted senses:", str(mispredicted))
    print("% of mispredicted senses:", str(mispredicted / len(x_test) * 100))

    return mispredicted
