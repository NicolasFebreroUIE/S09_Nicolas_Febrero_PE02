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
