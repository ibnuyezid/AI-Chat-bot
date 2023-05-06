from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np

nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sent, all_words):
    stemm = [stem(w) for w in tokenized_sent]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sent:
            bag[idx] = 1.0
    return bag


print([stem(w) for w in ["organize", "organization", "organizes"]])
