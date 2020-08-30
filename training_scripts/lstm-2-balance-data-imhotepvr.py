#balance_data.py
import os
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

training_file_dir = '../training_data/raw-data/'

training_all = np.zeros(shape=(0, 2))
for filename in os.listdir(training_file_dir):
    filename = training_file_dir + filename
    d = np.load(filename)
    training_all = np.concatenate((training_all, d))

train_data = list(training_all)

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []
backs = []

shuffle(train_data)

for data in train_data:
    print(data)
    choice = data[1]

    if choice == [1,0,0,0]:
        lefts.append(data)
    elif choice == [0,1,0,0]:
        forwards.append(data)
    elif choice == [0,0,0,1]:
        rights.append(data)
    elif choice == [0,0,1,0]:
        backs.append(data)
    else:
        print('no matches')


forwards = forwards[:len(lefts)][:len(rights)][:len(backs)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]
backs = backs[:len(forwards)]

final_data = forwards + lefts + rights + backs
shuffle(final_data)

np.save('../training_data/clean-data/training_data.npy', final_data)

train_data = np.load('../training_data/clean-data/training_data.npy')

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))
