import os
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def mlp(x_train: list, x_test: list, y_train: list, y_test: list, ngram: int = 1, use_stoplist: bool = False, hls = (100,)) -> int:

    if (use_stoplist):
        stoplist_file = Path(os.getcwd()).joinpath('stoplist-english.txt')
        with open(stoplist_file, mode='r') as f:
            stoplist_string = f.read()
        stoplist = stoplist_string.split("\n")
        model = make_pipeline(TfidfVectorizer(stop_words=stoplist, ngram_range=(ngram, ngram)), MLPClassifier(max_iter=500, hidden_layer_sizes=hls))
    else: model = make_pipeline(TfidfVectorizer(ngram_range=(ngram, ngram)), MLPClassifier(max_iter=500, hidden_layer_sizes=hls))

    model.fit(x_train, y_train)

    y_pred = np.array(model.predict(x_test))
    y_test = np.array(y_test)

    mispredicted = sum(y_test != y_pred)

    print("MLP")
    print("Number of mispredicted senses:", str(mispredicted))
    print("% of mispredicted senses:", str(mispredicted / len(x_test) * 100))

    return mispredicted
