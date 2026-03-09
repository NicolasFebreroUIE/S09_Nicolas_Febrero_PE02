# For this exercise i will use the same imports that we saw in the practice sessions before
# The only differences is that i will add sequential, the Embedding, LSTM and Dense layers.
# I also use matplotlib to see if the model learns properly as we saw in the tutorial.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# I changed the way i was doing the other exercise as i Knew that it had no good results
# So i now decided to define the categories here so i can read the results better later.
# 0 is bearish, 1 is bullish and 2 is neutral. 
sentiments = {0: "Bearish (Down)", 1: "Bullish (Up)", 2: "Neutral"}

# As always, firstly i load the data from the csv files given with pandas (as always)...
train_data = pd.read_csv('sent_train.csv')
valid_data = pd.read_csv('sent_valid.csv')
print("Checking the first rows of my training data:") # I print the first rows just to check if everything is loaded correctly.
print(train_data.head()) # This is like a simple EDA of wath we are using, i saw this in an online resouce so i will use it.

# Now i extract the text and the label for the training process using .values to have them as np.
train_texts = train_data['text'].astype(str).values
train_labels = train_data['label'].values
valid_texts = valid_data['text'].astype(str).values # I do the same for the validation data. 
valid_labels = valid_data['label'].values

# So now i will set the basic parameters for the word representaton
# I chose 10000 for the vocabulary size and 100 for the length, this is a general/theorical quantity, no tweets are that long
#or with that amount of characters
vocab_size = 10000
max_length = 100
embedding_dim = 64
oov_token = "<OOV>" # This is the token for out of vocabulary words we saw it in class...

# Then the tokenizer to transform the numbers to tokens
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_texts) # I fit the tokenizer on the training texts.

# The text sentences will now be transformed into sequences of integers
train_sequences = tokenizer.texts_to_sequences(train_texts)
valid_sequences = tokenizer.texts_to_sequences(valid_texts)
