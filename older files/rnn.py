# what i was doing real quick before i realized you commited a torch thing LMNFAO sigh

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from keras.datasets import imdb
import numpy as np

# const for data preprocessing
max_length = 256  # max length of the sequences
padding_type = 'post'  # padding type for sequences shorter than the maximum length
vocab_size = 40000  # size of the vocabulary used in the Embedding layer

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# preprocess data
def preprocess_data(data):
    return pad_sequences(data, maxlen=max_length, padding=padding_type)
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Model architecture
def build_model(vocab_size, embedding_dim=32, hidden_units=16):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(hidden_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Build and compile 
model = build_model(vocab_size)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train and evaluate 
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels), verbose=2)
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

model.save('my_model.keras')
