from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

from bayes import bayes
from decision_tree import decision_tree
from random_forest import random_forest
from svm import svm
from mlp import mlp
from extraction import extract_sentences_from_file, separate_sentences, separate_sentences_words, stem

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
    stemmed_1w = separate_sentences_words(stemmed_w, 0)
    stemmed_3w = separate_sentences_words(stemmed_w, 1)
    stemmed_5w = separate_sentences_words(stemmed_w, 2)
    stemmed_7w = separate_sentences_words(stemmed_w, 3)
    

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
    x_train_1s, x_test_1s, y_train_1s, y_test_1s = train_test_split(
        stemmed_1w, senses, test_size = 0.2, random_state = 0)
    x_train_3s, x_test_3s, y_train_3s, y_test_3s = train_test_split(
        stemmed_3w, senses, test_size = 0.2, random_state = 0)
    x_train_5s, x_test_5s, y_train_5s, y_test_5s = train_test_split(
        stemmed_5w, senses, test_size = 0.2, random_state = 0)
    x_train_7s, x_test_7s, y_train_7s, y_test_7s = train_test_split(
        stemmed_7w, senses, test_size = 0.2, random_state = 0)
    # Whole sentence
    x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(
        stemmed_w, senses, test_size = 0.2, random_state = 0)


    # ----------
    # | GRAPHS |
    # ----------

    # print("Total test data:", len(x_test_w), "\n")

    # # GRAPH #1
    # # Evaluation of Naive Bayes in function of n-gram model and
    # # comparison between "words", "pos tags", "stemmed" and "stemmed and stoplist" contextual info
    # # Words
    # bayes_1w = round(bayes(x_train_1w, x_test_1w, y_train_1w, y_test_1w) / len(x_test_w) * 100, 2)
    # bayes_3w = round(bayes(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)
    # bayes_5w = round(bayes(x_train_5w, x_test_5w, y_train_5w, y_test_5w) / len(x_test_w) * 100, 2)
    # bayes_7w = round(bayes(x_train_7w, x_test_7w, y_train_7w, y_test_7w) / len(x_test_w) * 100, 2)
    # bayes_w  = round(bayes(x_train_w, x_test_w, y_train_w, y_test_w) / len(x_test_w) * 100, 2)
    # # POS tags
    # bayes_1t = round(bayes(x_train_1t, x_test_1t, y_train_1t, y_test_1t, 1) / len(x_test_w) * 100, 2)
    # bayes_3t = round(bayes(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    # bayes_5t = round(bayes(x_train_5t, x_test_5t, y_train_5t, y_test_5t, 5) / len(x_test_w) * 100, 2)
    # bayes_7t = round(bayes(x_train_7t, x_test_7t, y_train_7t, y_test_7t, 7) / len(x_test_w) * 100, 2)
    # bayes_t  = round(bayes(x_train_t, x_test_t, y_train_t, y_test_t, 100) / len(x_test_w) * 100, 2)
    # # Stemmed
    # bayes_1s = round(bayes(x_train_1s, x_test_1s, y_train_1s, y_test_1s) / len(x_test_w) * 100, 2)
    # bayes_3s = round(bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)
    # bayes_5s = round(bayes(x_train_5s, x_test_5s, y_train_5s, y_test_5s) / len(x_test_w) * 100, 2)
    # bayes_7s = round(bayes(x_train_7s, x_test_7s, y_train_7s, y_test_7s) / len(x_test_w) * 100, 2)
    # bayes_s  = round(bayes(x_train_s, x_test_s, y_train_s, y_test_s) / len(x_test_w) * 100, 2)
    # # Stemmed and stoplist
    # bayes_1ss = round(bayes(x_train_1s, x_test_1s, y_train_1s, y_test_1s, use_stoplist=True) / len(x_test_w) * 100, 2)
    # bayes_3ss = round(bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)
    # bayes_5ss = round(bayes(x_train_5s, x_test_5s, y_train_5s, y_test_5s, use_stoplist=True) / len(x_test_w) * 100, 2)
    # bayes_7ss = round(bayes(x_train_7s, x_test_7s, y_train_7s, y_test_7s, use_stoplist=True) / len(x_test_w) * 100, 2)
    # bayes_ss  = round(bayes(x_train_s, x_test_s, y_train_s, y_test_s, use_stoplist=True) / len(x_test_w) * 100, 2)
    
    # g1_labels = ['1-gram', '3-gram', '5-gram', '7-gram', 'Complète']
    # g1_words = [bayes_1w, bayes_3w, bayes_5w, bayes_7w, bayes_w]
    # g1_tags = [bayes_1t, bayes_3t, bayes_5t, bayes_7t, bayes_t]
    # g1_stemmed = [bayes_1s, bayes_3s, bayes_5s, bayes_7s, bayes_s]
    # g1_stemmedstop = [bayes_1ss, bayes_3ss, bayes_5ss, bayes_7ss, bayes_ss]

    # g1_x = np.arange(len(g1_labels))  # the label locations
    # g1_width = 0.23  # the width of the bars

    # g1_fig, g1_ax = plt.subplots()
    # g1_rects_w = g1_ax.bar(g1_x - g1_width*1.5, g1_words, g1_width, label='Mots')
    # g1_rects_t = g1_ax.bar(g1_x - g1_width/2, g1_tags, g1_width, label='Étiquettes') # POS tags
    # g1_rects_s = g1_ax.bar(g1_x + g1_width/2, g1_stemmed, g1_width, label='Tronqués')
    # g1_rects_ss = g1_ax.bar(g1_x + g1_width*1.5, g1_stemmedstop, g1_width, label='Tronqués,\nsans mots outils') # POS tags

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g1_ax.set_ylabel('Taux d\'erreur (%)')
    # g1_ax.set_title('Taux d\'erreur de Naive Bayes selon\nl\'information contextuelle et n-gram')
    # g1_ax.set_xticks(g1_x, g1_labels)
    # g1_ax.legend()

    # g1_ax.bar_label(g1_rects_w,  padding=3, fontsize="x-small")
    # g1_ax.bar_label(g1_rects_t,  padding=3, fontsize="x-small")
    # g1_ax.bar_label(g1_rects_s,  padding=3, fontsize="x-small")
    # g1_ax.bar_label(g1_rects_ss, padding=3, fontsize="x-small")

    # g1_fig.tight_layout()
    # plt.show()


    # # GRAPH #2
    # # Comparison of each algorithm on 3-gram model for "words" contextual info
    # bayes_3w  = round(bayes(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)
    # tree_3w   = round(decision_tree(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)
    # forest_3w = round(random_forest(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)
    # svm_3w    = round(svm(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)
    # mlp_3w    = round(mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)

    # g2_labels = ['Naive Bayes', 'Arbre décision', 'Forêt aléatoire', 'SVM', 'MLP']
    # g2_words = [bayes_3w, tree_3w, forest_3w, svm_3w, mlp_3w]

    # g2_width = 0.35  # the width of the bars

    # g2_fig, g2_ax = plt.subplots()
    # g2_rects1 = g2_ax.bar(g2_labels, g2_words, g2_width)

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g2_ax.set_ylabel('Taux d\'erreur (%)')
    # g2_ax.set_title('Taux d\'erreur sur les mots comme information contextuelle\nde 3-grams selon l\'algorithme')
    # g2_ax.bar_label(g2_rects1, padding=3)
    
    # g2_fig.tight_layout()
    # plt.show()
    
    # # GRAPH #3
    # # Comparison of each algorithm on 3-gram model for "pos tags" contextual info
    # bayes_3t  = round(bayes(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    # tree_3t   = round(decision_tree(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    # forest_3t = round(random_forest(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    # svm_3t    = round(svm(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    # mlp_3t    = round(mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    
    # g3_labels = ['Naive Bayes', 'Arbre décision', 'Forêt aléatoire', 'SVM', 'MLP']
    # g3_words = [bayes_3t, tree_3t, forest_3t, svm_3t, mlp_3t]
    # bar_colors = ['tab:orange', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:orange']

    # g3_width = 0.35  # the width of the bars

    # g3_fig, g3_ax = plt.subplots()
    # g3_rects1 = g3_ax.bar(g3_labels, g3_words, g3_width, color=bar_colors)

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g3_ax.set_ylabel('Taux d\'erreur (%)')
    # g3_ax.set_title('Taux d\'erreur sur les étiquettes comme information contextuelle\nde 3-grams selon l\'algorithme')
    # g3_ax.bar_label(g3_rects1, padding=3)
    
    # g3_fig.tight_layout()
    # plt.show()
    
    # # GRAPH #4
    # # Comparison of each algorithm on 3-gram model for "stemmed words" contextual info
    # bayes_3s  = round(bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)
    # tree_3s   = round(decision_tree(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)
    # forest_3s = round(random_forest(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)
    # svm_3s    = round(svm(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)
    # mlp_3s    = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)

    # g4_labels = ['Naive Bayes', 'Arbre décision', 'Forêt aléatoire', 'SVM', 'MLP']
    # g4_words = [bayes_3s, tree_3s, forest_3s, svm_3s, mlp_3s]
    # bar_colors = ['tab:green', 'tab:green', 'tab:green', 'tab:green', 'tab:green']

    # g4_width = 0.35  # the width of the bars

    # g4_fig, g4_ax = plt.subplots()
    # g4_rects1 = g4_ax.bar(g4_labels, g4_words, g4_width, color=bar_colors)

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g4_ax.set_ylabel('Taux d\'erreur (%)')
    # g4_ax.set_title('Taux d\'erreur sur les mots tronqués comme information contextuelle\nde 3-grams selon l\'algorithme')
    # g4_ax.bar_label(g4_rects1, padding=3)
    
    # g4_fig.tight_layout()
    # plt.show()

    
    # # GRAPH #5
    # # Comparison of each algorithm on 3-gram model for "no stopwords" contextual info
    # bayes_3ss  = round(bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)
    # tree_3ss   = round(decision_tree(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)
    # forest_3ss = round(random_forest(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)
    # svm_3ss    = round(svm(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)
    # mlp_3ss    = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)

    # g5_labels = ['Naive Bayes', 'Arbre décision', 'Forêt aléatoire', 'SVM', 'MLP']
    # g5_words = [bayes_3ss, tree_3ss, forest_3ss, svm_3ss, mlp_3ss]
    # bar_colors = ['tab:red', 'tab:red', 'tab:red', 'tab:red', 'tab:red']

    # g5_width = 0.35  # the width of the bars

    # g5_fig, g5_ax = plt.subplots()
    # g5_rects1 = g5_ax.bar(g5_labels, g5_words, g5_width, color=bar_colors)

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g5_ax.set_ylabel('Taux d\'erreur (%)')
    # g5_ax.set_title('Taux d\'erreur sur les mots tronqués sans mots outils comme\ninformation contextuelle de 3-grams selon l\'algorithme')
    # g5_ax.bar_label(g5_rects1, padding=3)
    
    # g5_fig.tight_layout()
    # plt.show()

    # # GRAPH #6
    # # Comparison of each contextual info on 3-gram model for Naive Bayes algorithm
    # bayes_3w  = round(bayes(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)
    # bayes_3t  = round(bayes(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    # bayes_3s  = round(bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)
    # bayes_3ss = round(bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)
    
    # g6_labels = ['Mots', 'Catégories', 'Mots tronqués', 'Mots tronqués\nsans mots outils']
    # g6_words = [bayes_3w, bayes_3t, bayes_3s, bayes_3ss]
    # # bar_colors = ['tab:red', 'tab:red', 'tab:red', 'tab:red']

    # g6_width = 0.35  # the width of the bars

    # g6_fig, g6_ax = plt.subplots()
    # # g6_rects1 = g6_ax.bar(g6_labels, g6_words, g6_width, color=bar_colors)
    # g6_rects1 = g6_ax.bar(g6_labels, g6_words, g6_width)

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g6_ax.set_ylabel('Taux d\'erreur (%)')
    # g6_ax.set_title('Taux d\'erreur de Naive Bayes 3-gram selon l\'information contextuelle')
    # g6_ax.bar_label(g6_rects1, padding=3)
    
    # g6_fig.tight_layout()
    # plt.show()

    # # GRAPH #7
    # # Comparison of each contextual info on 3-gram model for Decision tree algorithm
    # tree_3w  = round(decision_tree(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)
    # tree_3t  = round(decision_tree(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    # tree_3s  = round(decision_tree(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)
    # tree_3ss = round(decision_tree(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)

    # g7_labels = ['Mots', 'Catégories', 'Mots tronqués', 'Mots tronqués\nsans mots outils']
    # g7_words = [tree_3w, tree_3t, tree_3s, tree_3ss]
    # bar_colors = ['tab:orange', 'tab:orange', 'tab:orange', 'tab:orange']

    # g7_width = 0.35  # the width of the bars

    # g7_fig, g7_ax = plt.subplots()
    # g7_rects1 = g7_ax.bar(g7_labels, g7_words, g7_width, color=bar_colors)

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g7_ax.set_ylabel('Taux d\'erreur (%)')
    # g7_ax.set_title('Taux d\'erreur de l\'arbre de décision 3-gram selon\nl\'information contextuelle')
    # g7_ax.bar_label(g7_rects1, padding=3)
    
    # g7_fig.tight_layout()
    # plt.show()


    # # GRAPH #8
    # # Comparison of each contextual info on 3-gram model for Random forest algorithm
    # forest_3w  = round(random_forest(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)
    # forest_3t  = round(random_forest(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    # forest_3s  = round(random_forest(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)
    # forest_3ss = round(random_forest(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)

    # g8_labels = ['Mots', 'Catégories', 'Mots tronqués', 'Mots tronqués\nsans mots outils']
    # g8_words = [forest_3w, forest_3t, forest_3s, forest_3ss]
    # bar_colors = ['tab:green', 'tab:green', 'tab:green', 'tab:green']

    # g8_width = 0.35  # the width of the bars

    # g8_fig, g8_ax = plt.subplots()
    # g8_rects1 = g8_ax.bar(g8_labels, g8_words, g8_width, color=bar_colors)

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g8_ax.set_ylabel('Taux d\'erreur (%)')
    # g8_ax.set_title('Taux d\'erreur de la forêt aléatoire 3-gram selon\nl\'information contextuelle')
    # g8_ax.bar_label(g8_rects1, padding=3)
    
    # g8_fig.tight_layout()
    # plt.show()

    # # GRAPH #9
    # # Comparison of each contextual info on 3-gram model for SVM algorithm
    # svm_3w  = round(svm(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)
    # svm_3t  = round(svm(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    # svm_3s  = round(svm(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)
    # svm_3ss = round(svm(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)

    # g9_labels = ['Mots', 'Catégories', 'Mots tronqués', 'Mots tronqués\nsans mots outils']
    # g9_words = [svm_3w, svm_3t, svm_3s, svm_3ss]
    # bar_colors = ['tab:red', 'tab:red', 'tab:red', 'tab:red']

    # g9_width = 0.35  # the width of the bars

    # g9_fig, g9_ax = plt.subplots()
    # g9_rects1 = g9_ax.bar(g9_labels, g9_words, g9_width, color=bar_colors)

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g9_ax.set_ylabel('Taux d\'erreur (%)')
    # g9_ax.set_title('Taux d\'erreur du SVM 3-gram selon l\'information contextuelle')
    # g9_ax.bar_label(g9_rects1, padding=3)
    
    # g9_fig.tight_layout()
    # plt.show()


    # # GRAPH #10
    # # Comparison of each contextual info on 3-gram model for MLP algorithm
    # mlp_3w  = round(mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w) / len(x_test_w) * 100, 2)
    # mlp_3t  = round(mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3) / len(x_test_w) * 100, 2)
    # mlp_3s  = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s) / len(x_test_w) * 100, 2)
    # mlp_3ss = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True) / len(x_test_w) * 100, 2)
    
    # g10_labels = ['Mots', 'Catégories', 'Mots tronqués', 'Mots tronqués\nsans mots outils']
    # g10_words = [mlp_3w, mlp_3t, mlp_3s, mlp_3ss]
    # bar_colors = ['tab:purple', 'tab:purple', 'tab:purple', 'tab:purple']

    # g10_width = 0.35  # the width of the bars

    # g10_fig, g10_ax = plt.subplots()
    # g10_rects1 = g10_ax.bar(g10_labels, g10_words, g10_width, color=bar_colors)

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g10_ax.set_ylabel('Taux d\'erreur (%)')
    # g10_ax.set_title('Taux d\'erreur du MLP 3-gram selon l\'information contextuelle')
    # g10_ax.bar_label(g10_rects1, padding=3)
    
    # g10_fig.tight_layout()
    # plt.show()


    # # GRAPH #11
    # # Comparison of different number of hidden layer neurons on 3-gram model for MLP algorithm
    # # Words
    # mlp_020w = round(mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w, hls=(20 ,)) / len(x_test_w) * 100, 2)
    # mlp_050w = round(mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w, hls=(50 ,)) / len(x_test_w) * 100, 2)
    # mlp_100w = round(mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w, hls=(100,)) / len(x_test_w) * 100, 2)
    # mlp_150w = round(mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w, hls=(150,)) / len(x_test_w) * 100, 2)
    # mlp_200w = round(mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w, hls=(200,)) / len(x_test_w) * 100, 2)
    # # POS tags
    # mlp_020t = round(mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3, hls=(20 ,)) / len(x_test_w) * 100, 2)
    # mlp_050t = round(mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3, hls=(50 ,)) / len(x_test_w) * 100, 2)
    # mlp_100t = round(mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3, hls=(100,)) / len(x_test_w) * 100, 2)
    # mlp_150t = round(mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3, hls=(150,)) / len(x_test_w) * 100, 2)
    # mlp_200t = round(mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3, hls=(200,)) / len(x_test_w) * 100, 2)
    # # Stemmed
    # mlp_020s = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, hls=(20 ,)) / len(x_test_w) * 100, 2)
    # mlp_050s = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, hls=(50 ,)) / len(x_test_w) * 100, 2)
    # mlp_100s = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, hls=(100,)) / len(x_test_w) * 100, 2)
    # mlp_150s = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, hls=(150,)) / len(x_test_w) * 100, 2)
    # mlp_200s = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, hls=(200,)) / len(x_test_w) * 100, 2)
    # # Stemmed and stoplist
    # mlp_020ss = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True, hls=(20 ,)) / len(x_test_w) * 100, 2)
    # mlp_050ss = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True, hls=(50 ,)) / len(x_test_w) * 100, 2)
    # mlp_100ss = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True, hls=(100,)) / len(x_test_w) * 100, 2)
    # mlp_150ss = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True, hls=(150,)) / len(x_test_w) * 100, 2)
    # mlp_200ss = round(mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_stoplist=True, hls=(200,)) / len(x_test_w) * 100, 2)

    # g11_labels = ['20', '50', '100', '150', '200']
    # g11_words = [mlp_020w, mlp_050w, mlp_100w, mlp_150w, mlp_200w]
    # g11_tags = [mlp_020t, mlp_050t, mlp_100t, mlp_150t, mlp_200t]
    # g11_stemmed = [mlp_020s, mlp_050s, mlp_100s, mlp_150s, mlp_200s]
    # g11_stemmedstop = [mlp_020ss, mlp_050ss, mlp_100ss, mlp_150ss, mlp_200ss]

    # g11_x = np.arange(len(g11_labels))  # the label locations
    # g11_width = 0.23  # the width of the bars

    # g11_fig, g11_ax = plt.subplots()
    # g11_rects_w = g11_ax.bar(g11_x - g11_width*1.5, g11_words, g11_width, label='Mots')
    # g11_rects_t = g11_ax.bar(g11_x - g11_width/2, g11_tags, g11_width, label='Étiquettes') # POS tags
    # g11_rects_s = g11_ax.bar(g11_x + g11_width/2, g11_stemmed, g11_width, label='Tronqués')
    # g11_rects_ss = g11_ax.bar(g11_x + g11_width*1.5, g11_stemmedstop, g11_width, label='Tronqués,\nsans mots outils') # POS tags

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # g11_ax.set_ylabel('Taux d\'erreur (%)')
    # g11_ax.set_title('Taux d\'erreur du MLP selon l\'information contextuelle\net le nombre de neurones cachés')
    # g11_ax.set_xticks(g11_x, g11_labels)
    # g11_ax.legend()

    # g11_ax.bar_label(g11_rects_w,  padding=3, fontsize="x-small")
    # g11_ax.bar_label(g11_rects_t,  padding=3, fontsize="x-small")
    # g11_ax.bar_label(g11_rects_s,  padding=3, fontsize="x-small")
    # g11_ax.bar_label(g11_rects_ss, padding=3, fontsize="x-small")

    # g11_fig.tight_layout()
    # plt.show()

    return


if __name__ == '__main__':
    analyse()
