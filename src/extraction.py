import os
from pathlib import Path
from typing import List
import numpy as np
from typing import Tuple
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# Need to uncomment and run the first time it is used
# import nltk
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

