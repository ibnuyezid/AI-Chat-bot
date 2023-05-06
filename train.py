import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

all_word = sorted(set(all_word))
tags = sorted(set(tags))
x_train = []
y_train = []

for (pattern_sen, tag) in xy:
    bag = bag_of_words(pattern_sen, all_word)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


batch_size = 8
dataset = ChatDataset()

train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
