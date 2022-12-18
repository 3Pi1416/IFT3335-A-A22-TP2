from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def svm(x_train: list, y_train: list, x_test: list, y_test: list, context: str):

    model = make_pipeline(TfidfVectorizer(), LinearSVC())
    model.fit(x_train, y_train)

    y_pred = np.array(model.predict(x_test))
    y_test = np.array(y_test)

    print("SVM for", context)
    print("Total test data:", len(x_test))
    print("Number of mislabeled senses:", str(sum(y_test != y_pred)))
    print("-----------")
