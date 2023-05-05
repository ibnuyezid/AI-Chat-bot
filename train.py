import json
from nltk_utils import tokenize, stem, bag_of_words
with open('intents.json', "r") as f:
    intents = json.load(f)

all_word = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_word.extend(w)
        xy.append((w, tag))
ignore_words = ["?", "!", ".", ","]

all_word = [stem(w) for w in all_word if w not in ignore_words]

print(all_word)
