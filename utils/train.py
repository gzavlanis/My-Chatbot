import json
import numpy as np 
import tensorflow as tf
import pickle 

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot

with open('data/intents.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating = 'post', maxlen = max_len)

# The model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

epochs = 600
history = model.fit(padded_sequences, np.array(training_labels), epochs = epochs)
loss = history.history['loss']
accuracy = history.history['accuracy']
print(history.history)

pyplot.plot(loss, label= 'Loss') # plot the loss
pyplot.plot(accuracy, label = 'Accuracy')
pyplot.title('Training loss vs accuracy of the model')
pyplot.xlabel('epochs', fontsize= 18)
pyplot.ylabel('loss-accuracy', fontsize= 18)
pyplot.grid()
pyplot.legend()
pyplot.show()

# evaluate on test data
results = model.evaluate(padded_sequences, np.array(training_labels), verbose = 1)

model.save("chatbot.keras") # save the trained model

# save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)

# save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol = pickle.HIGHEST_PROTOCOL)