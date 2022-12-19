import os
from pathlib import Path
from typing import List
import numpy as np
from typing import Tuple
# import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt')

def extract_sentences_from_file() -> Tuple[list[str], list[str], list[str]]:
    words, tags, senses = [], [], []

    interest_file = Path(os.getcwd()).joinpath('interest.acl94.txt')
    with open(interest_file, mode='r') as f:
        interest_text = f.read()
    
    # Split lines with special character to get each sentence
    interest_text_sentences = interest_text.split("$$\n")

    for line in interest_text_sentences:
        sentence_w, sentence_t = [], []

        # Clean special characters
        line = line.replace("======================================", '')
        
        for element in line.split():
            if "[" in element or "]" in element or element == "": pass
            else:
                # Get word/pos tag tuple
                word_tag = element.split("/")
                word = word_tag[0]

                # Get the sense of the "interest" occurence
                if "interest_" in word or "interests_" in word:
                    sentence_w.append(word[:-2])
                    senses.append(word[-1])
                else: sentence_w.append(word)

                # Get pos tag associated
                if len(word_tag) == 1: sentence_t.append("MISC")
                else: sentence_t.append(word_tag[1])

        words.append(" ".join(sentence_w))
        tags.append(" ".join(sentence_t))
    return words, tags, senses


def separate_sentences(sentences_words: list[str], 
                       sentences_tags: list[str], 
                       ngram: int) -> Tuple[list[str], list[str]]:

    ngrams_words, ngrams_tags = [], []

    for sentence_w, sentence_t in zip(sentences_words, sentences_tags):
        ngram_words, ngram_tags = [], []
        words, tags = sentence_w.split(), sentence_t.split()

        # Get the index of the "interest" occurence
        if   "*interests" in words: interest = words.index("*interests")
        elif "*interest"  in words: interest = words.index("*interest")
        elif "interests"  in words: interest = words.index("interests")
        elif "interest"   in words: interest = words.index("interest")
        else:
            ngrams_words.append([])
            ngrams_tags.append([])
            pass
        
        before_n = max(0, interest - ngram)
        after_n  = min(len(words), interest + ngram + 1)

        # Get the words and pos tags around the "interest" occurence
        for index in range(before_n, after_n):
            ngram_words.append(words[index])
            ngram_tags.append(tags[index])

        ngrams_words.append(" ".join(ngram_words))
        ngrams_tags.append(" ".join(ngram_tags))

    return ngrams_words, ngrams_tags


def separate_sentences_words(sentences_words: list[str], 
                       ngram: int) -> list[str]:

    ngrams_words = []

    for sentence_w in sentences_words:
        ngram_words = []
        words = sentence_w.split()

        # Get the index of the "interest" occurence
        if   "*interests" in words: interest = words.index("*interests")
        elif "*interest"  in words: interest = words.index("*interest")
        elif "interests"  in words: interest = words.index("interests")
        elif "interest"   in words: interest = words.index("interest")
        else:
            ngrams_words.append([])
            pass
        
        before_n = max(0, interest - ngram)
        after_n  = min(len(words), interest + ngram + 1)

        # Get the words and pos tags around the "interest" occurence
        for index in range(before_n, after_n):
            ngram_words.append(words[index])

        ngrams_words.append(" ".join(ngram_words))

    return ngrams_words

def stem(sentences: list[str]) -> list[str]:
    stemmed_words = []
    porter = PorterStemmer()

    for sentence in sentences:
        words = word_tokenize(sentence)
        stemmed_sentence = []
        for word in words:
            stemmed_sentence.append(porter.stem(word))
        stemmed_words.append(" ".join(stemmed_sentence))
    
    return stemmed_words


def extract_text_from_file():
    cwd = Path(os.getcwd())
    interest_text_file = cwd.joinpath('interest.acl94.txt')

    with open(interest_text_file, mode='r') as open_file:
        # read everything
        interest_text = open_file.read()
    
    # clean from special character
    interest_text = interest_text.replace("======================================", '')
    # split line from special character
    intrest_text_split = interest_text.split("$$")

    # delete useless jump line
    intrest_text_split = [line.replace("\n", '') for line in intrest_text_split]

    interest_text_extracted = []
    for line in intrest_text_split:
        new_line = []

        for word in line.split():

            if word == "[" or word == "]" or word == "":
                pass
            else:
                word_split = word.split("/")

                if len(word_split[0]) > 8 and word_split[0][:8].lower() == "interest":
                    number_position = 9
                    # delete plurial like skeleton code show (in studium)
                    if word_split[0][8] == "s":
                        number_position = number_position + 1
                    if word_split[0][number_position].isnumeric():
                        new_line.append(("interest", word_split[1], word_split[0][number_position]))
                    else:
                        if len(word_split) == 1:
                            # special case like MGMNP
                            new_line.append((word_split[0], ""))
                        else:
                            new_line.append((word_split[0], word_split[1]))
                else:
                    if len(word_split) == 1:
                        # special case like MGMNP
                        new_line.append((word_split[0], ""))
                    else:
                        new_line.append((word_split[0], word_split[1]))

        interest_text_extracted.append(new_line)
    return interest_text_extracted


def create_word_package(list_of_line, before: int, after: int, list_of_words_to_skip: List = []):
    list_of_package = [[], []]
    for line in list_of_line:
        size_line = len(line)
        for position_word in range(len(line)):

            word = line[position_word]
            if len(word) == 3:
                list_of_word = []
                before_position = max(0, position_word - before)
                after_position = min(size_line, position_word + after + 1)
                for i in range(before_position, position_word):
                    list_of_word.append(line[i][0])

                for i in range(position_word + 1, after_position):
                    list_of_word.append(line[i][0])

                list_of_package[0].append(np.asarray(list_of_word, dtype=object))
                list_of_package[1].append(word[2])

    return list_of_package


def create_syntax_package(list_of_line, before: int, after: int):
    list_of_package = [[], []]
    for line in list_of_line:
        size_line = len(line)
        for position_word in range(len(line)):

            word = line[position_word]
            if len(word) == 3:
                list_of_word = []
                before_position = max(0, position_word - before)
                after_position = min(size_line, position_word + after + 1)
                for i in range(before_position, position_word):
                    list_of_word.append(line[i][1])

                for i in range(position_word + 1, after_position):
                    list_of_word.append(line[i][1])

                list_of_package[0].append(np.asarray(list_of_word, dtype=object))
                list_of_package[1].append(word[2])

    return list_of_package