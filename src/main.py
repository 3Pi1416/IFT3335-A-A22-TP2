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
    # -----------------
    # | PRETREATEMENT |
    # -----------------

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
    # b_1w = bayes(x_train_1w, x_test_1w, y_train_1w, y_test_1w)
    # b_3w = bayes(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    # b_5w = bayes(x_train_5w, x_test_5w, y_train_5w, y_test_5w)
    # b_7w = bayes(x_train_7w, x_test_7w, y_train_7w, y_test_7w)
    # b_w  = bayes(x_train_w, x_test_w, y_train_w, y_test_w)
    # # POS tags
    # b_1t = bayes(x_train_1t, x_test_1t, y_train_1t, y_test_1t, 1)
    # b_3t = bayes(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    # b_5t = bayes(x_train_5t, x_test_5t, y_train_5t, y_test_5t, 5)
    # b_7t = bayes(x_train_7t, x_test_7t, y_train_7t, y_test_7t, 7)
    # b_t  = bayes(x_train_t, x_test_t, y_train_t, y_test_t, 100)
    # # Stemmed
    # b_1s = bayes(x_train_1s, x_test_1s, y_train_1s, y_test_1s)
    # b_3s = bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # b_5s = bayes(x_train_5s, x_test_5s, y_train_5s, y_test_5s)
    # b_7s = bayes(x_train_7s, x_test_7s, y_train_7s, y_test_7s)
    # b_s  = bayes(x_train_s, x_test_s, y_train_s, y_test_s)
    # # Stemmed and stoplist
    # b_1ss = bayes(x_train_1s, x_test_1s, y_train_1s, y_test_1s, use_sl=True)
    # b_3ss = bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)
    # b_5ss = bayes(x_train_5s, x_test_5s, y_train_5s, y_test_5s, use_sl=True)
    # b_7ss = bayes(x_train_7s, x_test_7s, y_train_7s, y_test_7s, use_sl=True)
    # b_ss  = bayes(x_train_s, x_test_s, y_train_s, y_test_s, use_sl=True)
    
    # g1_w = list(map(lambda w: round(w, 2), [b_1w, b_3w, b_5w, b_7w, b_w]))
    # g1_t = list(map(lambda t: round(t, 2), [b_1t, b_3t, b_5t, b_7t, b_t]))
    # g1_s = list(map(lambda s: round(s, 2), [b_1s, b_3s, b_5s, b_7s, b_s]))
    # g1_ss = list(map(lambda ss: round(ss, 2), [b_1ss, b_3ss, b_5ss, b_7ss, b_ss]))
    
    # make_multi_graph([g1_w, g1_t, g1_s, g1_ss], 1)


    # # GRAPH #2
    # # Comparison of each algorithm on 3-gram model for "words" contextual info
    # b_3w = bayes(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    # t_3w = decision_tree(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    # f_3w = random_forest(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    # s_3w = svm(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    # m_3w = mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    
    # g2_words = list(map(lambda w: round(w, 2), [b_3w, t_3w, f_3w, s_3w, m_3w]))

    # make_graph(g2_words, 2)

    
    # # GRAPH #3
    # # Comparison of each algorithm on 3-gram model for "pos tags" contextual info
    # b_3t = bayes(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    # t_3t = decision_tree(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    # f_3t = random_forest(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    # s_3t = svm(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    # m_3t = mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    
    # g3_t = list(map(lambda t: round(t, 2), [b_3t, t_3t, f_3t, s_3t, m_3t]))
    
    # make_graph(g3_t, 3, "tab:orange")

    
    # # GRAPH #4
    # # Comparison of each algorithm on 3-gram model for "stemmed words" contextual info
    # b_3s = bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # t_3s = decision_tree(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # f_3s = random_forest(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # s_3s = svm(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # m_3s = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s)

    # g4_s = list(map(lambda s: round(s, 2), [b_3s, t_3s, f_3s, s_3s, m_3s]))
    
    # make_graph(g4_s, 4, "tab:green")

    
    # # GRAPH #5
    # # Comparison of each algorithm on 3-gram model for "no stopwords" contextual info
    # b_3ss = bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)
    # t_3ss = decision_tree(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)
    # f_3ss = random_forest(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)
    # s_3ss = svm(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)
    # m_3ss = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)

    # g5_ss = list(map(lambda ss: round(ss, 2), [b_3ss, t_3ss, f_3ss, s_3ss, m_3ss]))
    
    # make_graph(g5_ss, 5, "tab:red")


    # # GRAPH #6
    # # Comparison of each contextual info on 3-gram model for Naive Bayes algorithm
    # b_3w  = bayes(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    # b_3t  = bayes(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    # b_3s  = bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # b_3ss = bayes(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)
    
    # g6_b = list(map(lambda b: round(b, 2), [b_3w, b_3t, b_3s, b_3ss]))
    
    # make_graph(g6_b, 6)


    # # GRAPH #7
    # # Comparison of each contextual info on 3-gram model for Decision tree algorithm
    # t_3w  = decision_tree(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    # t_3t  = decision_tree(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    # t_3s  = decision_tree(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # t_3ss = decision_tree(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)
    
    # g7_t = list(map(lambda t: round(t, 2), [t_3w, t_3t, t_3s, t_3ss]))
    
    # make_graph(g7_t, 7, "tab:orange")


    # # GRAPH #8
    # # Comparison of each contextual info on 3-gram model for Random forest algorithm
    # f_3w  = random_forest(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    # f_3t  = random_forest(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    # f_3s  = random_forest(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # f_3ss = random_forest(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)
    
    # g8_f = list(map(lambda f: round(f, 2), [f_3w, f_3t, f_3s, f_3ss]))
    
    # make_graph(g8_f, 8, "tab:green")


    # # GRAPH #9
    # # Comparison of each contextual info on 3-gram model for SVM algorithm
    # s_3w  = svm(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    # s_3t  = svm(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    # s_3s  = svm(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # s_3ss = svm(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)

    # g9_s = list(map(lambda s: round(s, 2), [s_3w, s_3t, s_3s, s_3ss]))
    
    # make_graph(g9_s, 9, "tab:red")


    # # GRAPH #10
    # # Comparison of each contextual info on 3-gram model for MLP algorithm
    # m_3w  = mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w)
    # m_3t  = mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3)
    # m_3s  = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s)
    # m_3ss = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True)
    
    # g10_m = list(map(lambda m: round(m, 2), [m_3w, m_3t, m_3s, m_3ss]))

    # make_graph(g10_m, 10, "tab:purple")


    # # GRAPH #11
    # # Comparison of different number of hidden layer neurons on 3-gram model for MLP algorithm
    # # Words
    # mlp_020w = mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w, hls=(20 ,))
    # mlp_050w = mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w, hls=(50 ,))
    # mlp_100w = mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w, hls=(100,))
    # mlp_150w = mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w, hls=(150,))
    # mlp_200w = mlp(x_train_3w, x_test_3w, y_train_3w, y_test_3w, hls=(200,))
    # # POS tags
    # mlp_020t = mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3, hls=(20 ,))
    # mlp_050t = mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3, hls=(50 ,))
    # mlp_100t = mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3, hls=(100,))
    # mlp_150t = mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3, hls=(150,))
    # mlp_200t = mlp(x_train_3t, x_test_3t, y_train_3t, y_test_3t, 3, hls=(200,))
    # # Stemmed
    # mlp_020s = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, hls=(20 ,))
    # mlp_050s = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, hls=(50 ,))
    # mlp_100s = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, hls=(100,))
    # mlp_150s = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, hls=(150,))
    # mlp_200s = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, hls=(200,))
    # # Stemmed and stoplist
    # mlp_020ss = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True, hls=(20 ,))
    # mlp_050ss = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True, hls=(50 ,))
    # mlp_100ss = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True, hls=(100,))
    # mlp_150ss = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True, hls=(150,))
    # mlp_200ss = mlp(x_train_3s, x_test_3s, y_train_3s, y_test_3s, use_sl=True, hls=(200,))

    # g11_w = list(map(lambda w: round(w, 2), [mlp_020w, mlp_050w, mlp_100w, mlp_150w, mlp_200w]))
    # g11_t = list(map(lambda t: round(t, 2), [mlp_020t, mlp_050t, mlp_100t, mlp_150t, mlp_200t]))
    # g11_s = list(map(lambda s: round(s, 2), [mlp_020s, mlp_050s, mlp_100s, mlp_150s, mlp_200s]))
    # g11_ss = list(map(lambda ss: round(ss, 2), [mlp_020ss, mlp_050ss, mlp_100ss, mlp_150ss, mlp_200ss]))

    # make_multi_graph([g11_w, g11_t, g11_s, g11_ss], 11)

    return

def make_graph(graph_data: list[int], graph_nbr: int = 2, bar_colors: str = "tab:blue"):
    match graph_nbr:
        case 2:
            labels = ['Naive Bayes', 'Arbre décision', 'Forêt aléatoire', 'SVM', 'MLP']
            title = "Taux d\'erreur sur les mots comme information contextuelle\nde 3-grams selon l\'algorithme"
        case 3:
            labels = ['Naive Bayes', 'Arbre décision', 'Forêt aléatoire', 'SVM', 'MLP']
            title = "Taux d\'erreur sur les étiquettes comme information contextuelle\nde 3-grams selon l\'algorithme"
        case 4:
            labels = ['Naive Bayes', 'Arbre décision', 'Forêt aléatoire', 'SVM', 'MLP']
            title = "Taux d\'erreur sur les mots tronqués comme information contextuelle\nde 3-grams selon l\'algorithme"
        case 5:
            labels = ['Naive Bayes', 'Arbre décision', 'Forêt aléatoire', 'SVM', 'MLP']
            title = "Taux d\'erreur sur les mots tronqués sans mots outils comme\ninformation contextuelle de 3-grams selon l\'algorithme"
        case 6:
            labels = ['Mots', 'Catégories', 'Mots tronqués', 'Mots tronqués\nsans mots outils']
            title = "Taux d\'erreur de Naive Bayes 3-gram selon l\'information contextuelle"
        case 7:
            labels = ['Mots', 'Catégories', 'Mots tronqués', 'Mots tronqués\nsans mots outils']
            title = "Taux d\'erreur de l\'arbre de décision 3-gram selon\nl\'information contextuelle"
        case 8:
            labels = ['Mots', 'Catégories', 'Mots tronqués', 'Mots tronqués\nsans mots outils']
            title = "Taux d\'erreur de la forêt aléatoire 3-gram selon\nl\'information contextuelle"
        case 9:
            labels = ['Mots', 'Catégories', 'Mots tronqués', 'Mots tronqués\nsans mots outils']
            title = "Taux d\'erreur du SVM 3-gram selon l\'information contextuelle"
        case 10:
            labels = ['Mots', 'Catégories', 'Mots tronqués', 'Mots tronqués\nsans mots outils']
            title = "Taux d\'erreur du MLP 3-gram selon l\'information contextuelle"
        case _:
            labels = ['Naive Bayes', 'Arbre décision', 'Forêt aléatoire', 'SVM', 'MLP']
            title = "Taux d\'erreur sur les mots comme information contextuelle\nde 3-grams selon l\'algorithme"

    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(labels, graph_data, width, color=bar_colors)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Taux d\'erreur (%)')
    ax.set_title(title)
    ax.bar_label(rects1, padding=3)
    
    fig.tight_layout()
    plt.show()

def make_multi_graph(graph_data: list[list[int]], graph_nbr: int = 1):

    if graph_nbr == 1:
        labels = ['1-gram', '3-gram', '5-gram', '7-gram', 'Complète']
        title = "Taux d\'erreur de Naive Bayes selon\nl\'information contextuelle et n-gram"
    else: 
        labels = ['20', '50', '100', '150', '200']
        title = "Taux d\'erreur du MLP selon l\'information contextuelle\net le nombre de neurones cachés"

    g_x = np.arange(len(labels))  # the label locations
    g_width = 0.23  # the width of the bars

    g_fig, g_ax = plt.subplots()
    g_rects_w = g_ax.bar(g_x - g_width*1.5, graph_data[0], g_width, label='Mots')
    g_rects_t = g_ax.bar(g_x - g_width/2, graph_data[1], g_width, label='Étiquettes')
    g_rects_s = g_ax.bar(g_x + g_width/2, graph_data[2], g_width, label='Tronqués')
    g_rects_ss = g_ax.bar(g_x + g_width*1.5, graph_data[3], g_width, label='Tronqués,\nsans mots outils')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    g_ax.set_ylabel('Taux d\'erreur (%)')
    g_ax.set_title(title)
    g_ax.set_xticks(g_x, labels)
    g_ax.legend()

    g_ax.bar_label(g_rects_w,  padding=3, fontsize="x-small")
    g_ax.bar_label(g_rects_t,  padding=3, fontsize="x-small")
    g_ax.bar_label(g_rects_s,  padding=3, fontsize="x-small")
    g_ax.bar_label(g_rects_ss, padding=3, fontsize="x-small")

    g_fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    analyse()
