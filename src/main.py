from sklearn.model_selection import train_test_split

from src.extraction import extract_text_from_file, create_word_package, create_syntax_package


def annalyse():
    extract_from_text = extract_text_from_file()
    word_package = create_word_package(extract_from_text, 2, 2)
    syntax_package = create_syntax_package(extract_from_text, 2, 2)

    X_train, X_test, y_train, y_test = train_test_split(word_package[0], word_package[1], test_size=0.5, random_state=0)

    return


if __name__ == '__main__':
    annalyse()
