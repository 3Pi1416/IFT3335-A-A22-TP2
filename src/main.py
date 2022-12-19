from sklearn.model_selection import train_test_split

# Pour Hugo
# from src.bayes import bayes
# from src.extraction import extract_text_from_file, create_word_package, create_syntax_package

# Pour Doum
from bayes import bayes
from decision_tree import decision_tree
from random_forest import random_forest
from svm import svm
from mlp import mlp
from extraction import extract_sentences_from_file, separate_sentences, separate_sentences_words, stem

# Value of n for n-gram model
ngram = 2

def analyse():
    # --------------
    # | EXTRACTION |
    # --------------

    # sentences_w: Array of whole sentences of words
    # sentences_t: Array of whole sentences of associated pos tags
    # senses: Array of senses associated to the "interest" occurence
    sentences_w, sentences_t, senses = extract_sentences_from_file()

    # ngramsX_w: Array of sentences of words using n-gram model 
    # ngramsX_t: Array of associated pos tags using n-gram model
    # For X, the ngramsX model is (X-1) words before, (X-1) words after
    # The default model will be 5-gram
    ngrams1_w, ngrams1_t = separate_sentences(sentences_w, sentences_t, 0)
    ngrams3_w, ngrams3_t = separate_sentences(sentences_w, sentences_t, 1)
    ngrams5_w, ngrams5_t = separate_sentences(sentences_w, sentences_t, 2)
    ngrams7_w, ngrams7_t = separate_sentences(sentences_w, sentences_t, 3)

    # stemmed_w: Array of sentences of stemmed words (whole and 3-gram)
    stemmed_w = stem(sentences_w)
    stemmed_3w = separate_sentences_words(stemmed_w, 1)
    

    # -----------------------------
    # | TRAINING AND TESTING SETS |
    # -----------------------------

    # Contextual information: words
    x_train_1w, x_test_1w, y_train_1w, y_test_1w = train_test_split(
        ngrams1_w, senses, test_size = 0.2, random_state = 0)
    x_train_3w, x_test_3w, y_train_3w, y_test_3w = train_test_split(
        ngrams3_w, senses, test_size = 0.2, random_state = 0)
    x_train_5w, x_test_5w, y_train_5w, y_test_5w = train_test_split(
        ngrams5_w, senses, test_size = 0.2, random_state = 0)
    x_train_7w, x_test_7w, y_train_7w, y_test_7w = train_test_split(
        ngrams7_w, senses, test_size = 0.2, random_state = 0)
    # Whole sentence
    x_train_w, x_test_w, y_train_w, y_test_w = train_test_split(
        sentences_w, senses, test_size = 0.2, random_state = 0) 

    # Contextual information: pos tags
    x_train_1t, x_test_1t, y_train_1t, y_test_1t = train_test_split(
        ngrams1_t, senses, test_size = 0.2, random_state = 0)
    x_train_3t, x_test_3t, y_train_3t, y_test_3t = train_test_split(
        ngrams3_t, senses, test_size = 0.2, random_state = 0)
    x_train_5t, x_test_5t, y_train_5t, y_test_5t = train_test_split(
        ngrams5_t, senses, test_size = 0.2, random_state = 0)
    x_train_7t, x_test_7t, y_train_7t, y_test_7t = train_test_split(
        ngrams7_t, senses, test_size = 0.2, random_state = 0)
    # Whole sentence
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(
        sentences_t, senses, test_size = 0.2, random_state = 0)
    
    # Contextual information: stemmed words
    x_train_3s, x_test_3s, y_train_3s, y_test_3s = train_test_split(
        stemmed_3w, senses,
        test_size = 0.2, random_state = 0)


    # ----------
    # | GRAPHS |
    # ----------

    print("Total test data:", len(x_test_w), "\n")

    # GRAPH #1
    # Evaluation of Naive Bayes in function of n-gram model and
    # comparison between "words" and "pos tags" contextual info
    bayes_1w = bayes(x_train_1w, x_test_1w, y_train_1w, y_test_1w)
    bayes_3w = bayes(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    bayes_5w = bayes(x_train_5w, x_test_5w, y_train_5w, y_test_5w)
    bayes_7w = bayes(x_train_7w, x_test_7w, y_train_7w, y_test_7w)
    bayes_w  = bayes(x_train_w, x_test_w, y_train_w, y_test_w)
    bayes_1t = bayes(x_train_1t, x_test_1t, y_train_1t, y_test_1t)
    bayes_3t = bayes(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    bayes_5t = bayes(x_train_5t, x_test_5t, y_train_5t, y_test_5t)
    bayes_7t = bayes(x_train_7t, x_test_7t, y_train_7t, y_test_7t)
    bayes_t  = bayes(x_train_t, x_test_t, y_train_t, y_test_t)

    # GRAPH #2
    # Comparison of each algorithm on 3-gram model for "words" contextual info
    bayes_3w = bayes(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    tree_3w = decision_tree(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    forest_3w = random_forest(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    svm_3w = svm(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    mlp_3w = mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    
    # GRAPH #3
    # Comparison of each algorithm on 3-gram model for "pos tags" contextual info
    bayes_3t = bayes(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    tree_3t = decision_tree(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    forest_3t = random_forest(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    svm_3t = svm(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    mlp_3t = mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    
    # GRAPH #4
    # Comparison of each algorithm on 3-gram model for "stemmed words" contextual info
    bayes_3s = bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    tree_3s = decision_tree(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    forest_3s = random_forest(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    svm_3s = svm(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    mlp_3s = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    
    # GRAPH #5
    # Comparison of each algorithm on 3-gram model for "no stopwords" contextual info
    # TODO

    # GRAPH #6
    # Comparison of each contextual info on 3-gram model for Naive Bayes algorithm
    bayes_3w = bayes(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    bayes_3t = bayes(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    bayes_3s = bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # TODO no stopwords

    # GRAPH #6
    # Comparison of each contextual info on 3-gram model for Decision tree algorithm
    tree_3w = decision_tree(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    tree_3t = decision_tree(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    tree_3s = decision_tree(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # TODO no stopwords

    # GRAPH #7
    # Comparison of each contextual info on 3-gram model for Random forest algorithm
    forest_3w = random_forest(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    forest_3t = random_forest(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    forest_3s = random_forest(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # TODO no stopwords

    # GRAPH #8
    # Comparison of each contextual info on 3-gram model for SVM algorithm
    svm_3w = svm(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    svm_3t = svm(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    svm_3s = svm(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # TODO no stopwords

    # GRAPH #9
    # Comparison of each contextual info on 3-gram model for MLP algorithm
    mlp_3w = mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    mlp_3t = mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t)
    mlp_3s = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # TODO no stopwords

    # GRAPH #10
    # Comparison of different number of hidden layer neurons on 3-gram model for MLP algorithm
    # TODO


    return


if __name__ == '__main__':
    analyse()
