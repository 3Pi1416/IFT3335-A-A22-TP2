from sklearn.model_selection import train_test_split

from src.bayes import bayes
from src.extraction import extract_text_from_file, create_word_package, create_syntax_package


def annalyse():
    extract_from_text = extract_text_from_file()
    word_package = create_word_package(extract_from_text, 2, 2)
    syntax_package = create_syntax_package(extract_from_text, 2, 2)


 # need doom here
    new_for_test = [[],[]]
    for i in range(len(word_package[0])):
        if len(word_package[0][i]) == 4 :
            new_for_test[0].append(word_package[0][i])
            new_for_test[1].append(word_package[1][i])




    word_x_train, word_x_test, word_y_train, word_y_test = train_test_split(new_for_test[0],
                                                                            new_for_test[1],
                                                                            test_size=0.5,
                                                                            random_state=0)
    syntax_x_train, syntax_x_test, syntax_y_train, syntax_y_test = train_test_split(syntax_package[0],
                                                                                    syntax_package[1],
                                                                                    test_size=0.5, random_state=0)

    bayes(word_x_train, word_y_train, word_x_test, word_y_test)
    return


if __name__ == '__main__':
    annalyse()
