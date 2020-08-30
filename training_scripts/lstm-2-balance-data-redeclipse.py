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

lefts1 = []
lefts2 = []
rights1 = []
rights2 = []
forwards = []

shuffle(train_data)

for data in train_data:
    choice = data[1]

    if choice == [1,0,0,0,0]:
        lefts1.append(data)
    elif choice == [0,1,0,0,0]:
        lefts2.append(data)
    elif choice == [0,0,1,0,0]:
        forwards.append(data)
    elif choice == [0,0,0,1,0]:
        rights1.append(data)
    elif choice == [0,0,0,0,1]:
        rights2.append(data)
    else:
        print('no matches')


forwards = forwards[:len(lefts1)][:len(rights1)][:len(lefts2)][:len(rights2)]
lefts1 = lefts1[:len(forwards)]
rights1 = rights1[:len(forwards)]
lefts2 = lefts2[:len(forwards)]
rights2 = rights2[:len(forwards)]
#backs = backs[:len(forwards)]

final_data = forwards + lefts1 + rights1 + lefts2 + rights2
shuffle(final_data)

np.save('../training_data/clean-data/training_data.npy', final_data)

train_data = np.load('../training_data/clean-data/training_data.npy')

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))
