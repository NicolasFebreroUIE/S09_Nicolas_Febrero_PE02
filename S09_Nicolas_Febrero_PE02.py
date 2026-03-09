# For this exercise i will use the same imports that we saw in the practice sessions before, same as the ones in the las evaluated practice
# The only differencies is taht i will add sequential, the Embedding, LSTM and Dense layers.
# I also use re for some basic cleaning as i saw in an online tutorial.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import re
from sklearn.metrics import classification_report

# I changed the way i was doing the other exercise as i Knew that it had no good results
# So i now decided to define the categories here so i can read the results better later.
# 0 is bearish, 1 is bullish and 2 is neutral. 
sentiments = {0: "Bearish (Down)", 1: "Bullish (Up)", 2: "Neutral"}

# As always, firstly i load the data from the csv files given with pandas (as always)...
train_data = pd.read_csv('sent_train.csv')
valid_data = pd.read_csv('sent_valid.csv')
print("Checking the first rows ")
print(train_data.head()) 

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text) # I remove links
    text = re.sub(r'\$\w+', '', text)   # I remove stock tickers like $AAPL
    text = re.sub(r'[^a-zA-Z\s]', '', text) # I remove symbols and numbers
    return text

# Now i extract the text and the label for the training process using .values to have them as np.
train_texts = [basic_clean(t) for t in train_data['text']]
train_labels = train_data['label'].values
valid_texts = [basic_clean(t) for t in valid_data['text']]
valid_labels = valid_data['label'].values

# So now i will set the basic parameters for the word representaton
vocab_size = 5000
max_length = 50
embedding_dim = 64
oov_token = "<OOV>" 

# Then the tokenizer to transform the numbers to tokens
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_texts) 

# The text sentences will now be transformed into sequences of integers
train_sequences = tokenizer.texts_to_sequences(train_texts)
valid_sequences = tokenizer.texts_to_sequences(valid_texts)

# I add padding to make them all equall. I use 'pre' because i read it is better for LSTM.
train_p = pad_sequences(train_sequences, maxlen=max_length, padding='pre', truncating='pre')
valid_p = pad_sequences(valid_sequences, maxlen=max_length, padding='pre', truncating='pre')
train_l = np.array(train_labels)
valid_l = np.array(valid_labels)

# Build the neural network using the Sequential model.
model = Sequential()

# Embedding layer with masking to ignore the zeros.
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True))
# The LSTM layer manages the temporal nature and captures the context.
model.add(LSTM(128, dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Training process
print("training process:")
history = model.fit(
    train_p, 
    train_l, 
    epochs=5, 
    batch_size=32, 
    validation_data=(valid_p, valid_l)
)


# Instead of doing it "by hand" with a few sentences, i use the whole validation set.
print("FINAL EVALUATION ON VALIDATION SET:")
# I predict the probabilities for all tweets in the validation set.
predictions = model.predict(valid_p)
# I take the category with the highest probability.
predicted_classes = np.argmax(predictions, axis=1)

# I use classification_report to see the performance in each category.
report = classification_report(valid_l, predicted_classes, target_names=list(sentiments.values()))
print(report)

# I want to see the final accuracy result.
loss, acc = model.evaluate(valid_p, valid_l, verbose=0)
print(f"Final Validation Accuracy: {acc:.4f}")

# Plotting the accuracy and loss evolution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy through Epochs')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss through Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('my_training_plots.png')
print("Plots saved like in my_training_plots.png")

# REMARKS: At first, the accurac level was at 0.65, i had to search i way to fixed it beacuse it will think all frases were neutral
# Also i had to change the model i was trainign because i didnt use the tokenize function that owrked with the tweets symbols and charcaters.
# This is my Github link if needed: 