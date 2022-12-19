from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def decision_tree(x_train: list, x_test: list, y_train: list, y_test: list):

    model = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())
    model.fit(x_train, y_train)

    y_pred = np.array(model.predict(x_test))
    y_test = np.array(y_test)

    mislabeled = sum(y_test != y_pred)

    print("Decision tree")
    print("Number of mislabeled senses:", str(mislabeled))
    print("% of mislabeled senses:", str(mislabeled / len(x_test) * 100))

    return mislabeled
