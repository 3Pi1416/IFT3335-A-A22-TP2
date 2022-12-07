from sklearn.naive_bayes import MultinomialNB


def bayes(x_train, y_train, x_test, y_test):
    gnb = MultinomialNB()

    #f
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)


    # a changer !!!! TODO
    print("Number of mislabeled points out of a total %d points : %d", (x_test.shape[0], (y_test != y_pred).sum()))
