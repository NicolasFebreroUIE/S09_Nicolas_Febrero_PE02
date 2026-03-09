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
