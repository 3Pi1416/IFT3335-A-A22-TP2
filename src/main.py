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
from extraction import extract_sentences_from_file, separate_sentences

# Value of n for n-gram model
ngram = 2

def analyse():
    # sentences_w: Array of whole sentences of words
    # sentences_t: Array of whole sentences of associated pos tags
    # senses: Array of senses associated to the "interest" occurence
    sentences_w, sentences_t, senses = extract_sentences_from_file()

    # ngrams_w: Array of sentences of words using n-gram model
    # ngrams_t: Array of associated pos tags using n-gram model
    ngrams_w, ngrams_t = separate_sentences(sentences_w, sentences_t, ngram)
    
    # Training and testing sets for "words" contextual information
    x_train_w, x_test_w, y_train_w, y_test_w = train_test_split(
        ngrams_w, senses,
        test_size = 0.2, random_state = 0)

    # Training and testing sets for "pos tags" contextual information
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(
        ngrams_t, senses,
        test_size = 0.2, random_state = 0)

    # NAIVE BAYES
    bayes(x_train_w, y_train_w, x_test_w, y_test_w, "words")
    bayes(x_train_t, y_train_t, x_test_t, y_test_t, "pos tags")
    
    # DECISION TREE
    decision_tree(x_train_w, y_train_w, x_test_w, y_test_w, "words")
    decision_tree(x_train_t, y_train_t, x_test_t, y_test_t, "pos tags")
    
    # RANDOM FOREST
    random_forest(x_train_w, y_train_w, x_test_w, y_test_w, "words")
    random_forest(x_train_t, y_train_t, x_test_t, y_test_t, "pos tags")
    
    # SVM
    svm(x_train_w, y_train_w, x_test_w, y_test_w, "words")
    svm(x_train_t, y_train_t, x_test_t, y_test_t, "pos tags")
    
    # MLP
    mlp(x_train_w, y_train_w, x_test_w, y_test_w, "words")
    mlp(x_train_t, y_train_t, x_test_t, y_test_t, "pos tags")

    return


if __name__ == '__main__':
    analyse()
