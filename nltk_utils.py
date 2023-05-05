from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sent, all_words):
    pass


print([stem(w) for w in ["organize", "organization", "organizes"]])
