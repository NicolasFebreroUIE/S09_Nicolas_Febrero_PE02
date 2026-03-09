# For this exercise i will use the same imports that we saw in the practice sessions before, same as the ones in the las evaluated practice
# The only differencies is taht i will add sequential, the Embedding, LSTM and Dense layers.

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Here I load the train and valid datasets from the csv files given. 
train_data = pd.read_csv('sent_train.csv')
valid_data = pd.read_csv('sent_valid.csv')

# I separate the text and the labels for the training and validation process.
train_texts = train_data["text"].values #This will take all the values from the text column. 
train_labels = train_data["label"].values #This  take all the values from the label column.
valid_texts = valid_data["text"].values #This take all the values from the tex column.
valid_labels = valid_data["label"].values #This take all the values from the label column. (I just did this to add comments as requested, its easy, but helpful to follow the code)

# Here I set the basic parameters for the text processing following the general structure of the practice sessions and theorical content online

vocab_size = 10000 
max_length = 100 # I add this big number because is the typical and because it is hard to find a tweet longer than that in one sentence...
embedding_dim = 64 #General numbes for any text processing task, as long as it is not specified in the problem.

