from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def bayes(x_train: list, x_test: list, y_train: list, y_test: list) -> int:

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(x_train, y_train)

    y_pred = np.array(model.predict(x_test))
    y_test = np.array(y_test)

    mislabeled = sum(y_test != y_pred)

    print("Naive Bayes")
    print("Number of mislabeled senses:", str(mislabeled))
    print("% of mislabeled senses:", str(mislabeled / len(x_test) * 100))

    return mislabeled
