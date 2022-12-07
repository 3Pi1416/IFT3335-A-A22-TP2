import os
from pathlib import Path
from typing import List
import numpy as np


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
