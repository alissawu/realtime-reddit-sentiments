# what i was doing real quick before i realized you commited a torch thing LMNFAO sigh
from ensurepip import bootstrap

import tensorflow as tf
# noinspection PyUnresolvedReferences
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

# const for data preprocessing
max_length = 256  #2^8 max length of the sequences
padding_type = 'post'  # padding type for sequences shorter than the maximum length
vocab_size = 32768  # size of the vocabulary used in the Embedding layer
#5000,10000, 20000
# Load the IMDB dataset

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# preprocess data
def preprocess_data(data):
    return pad_sequences(data, maxlen=max_length, padding=padding_type)

# Manually split training data into training and validation sets
split_1 = 5/2
split_2 = 20/17
split_1_index = int(len(test_data) // split_1)
split_2_index = int(len(test_data) // split_2)+1

# Split the data
train_data, val_data, test_data = np.append(train_data,test_data[:split_1_index]), test_data[split_1_index:split_2_index],test_data[split_2_index:]
train_labels, val_labels,   test_labels = np.append(train_labels,test_labels[:split_1_index]), test_labels[split_1_index:split_2_index],test_labels[split_2_index:]


train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)
val_data = preprocess_data(val_data)

# Model architecture
# embedding_dim from 32 to 256
# i
def build_model(vocab_size, embedding_dim=64, hidden_units=16):
    model = Sequential([
        #inputs 256 x 1
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        # outputs 256 x 1 x embedding_dim
        GlobalAveragePooling1D(),
        #outputs 256 x embedding_data
        Dropout(0.25),
        # outputs 256 x embedding_data
        Dense(hidden_units, activation='relu'),

        Dense(1, activation='sigmoid')
    ])
    return model
def build_model_initial(vocab_size, embedding_dim=256, hidden_units=16):
    model = Sequential([

        Embedding(vocab_size, embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(hidden_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model
def build_model_secondary(vocab_size, embedding_dim=256, hidden_units=16):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(hidden_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model
print(f"{len(train_data)} train samples + {len(train_labels)} train labels")
print(f"{len(val_data)} validation samples + {len(val_labels)} validation labels")
print(f"{len(test_data)} test samples + {len(test_labels)} test labels")
# Build and compile
embedding_dimensions    =   [2**i for i in range(4,9)]
colors = ['red', 'orange', 'green', 'blue', 'purple']
#colors = ['red', 'green', 'blue', 'orange', 'purple']
emb_history =   {}
for i in embedding_dimensions:
    print(f"\nModel with embedding dimension {i}: ")

    model1 = build_model(vocab_size,i)
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model1.summary()

# Train and evaluate our model based on imdb review data ONLY - no reddit yet

    history = model1.fit(train_data, train_labels, epochs=7, batch_size=32, validation_data=(val_data, val_labels), verbose=2)
    emb_history[i] = history.history
    test_loss, test_acc = model1.evaluate(test_data, test_labels, verbose=2)

    print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

def boostrap_binary_cross_entropy(data, labels, n_iter):
    bootstrap_accuracy =   []
    bootstrap_loss = []
    samp_size    =   len(labels)
    for i in range(n_iter):
        ith_indexes =   np.random.choice(len(data),samp_size)
        data_boot   = data[ith_indexes]
        labels_boot = labels[ith_indexes]

        data_boot_pred  = np.clip(model.predict(data_boot),1e-7,-1e-7)
        accuracy    =   np.mean(data_boot_pred == labels_boot)
        loss        =   -1  * np.mean(labels_boot *   np.log(data_boot_pred) +   (1  -   labels_boot) * np.log(1 - data_boot_pred))
        bootstrap_accuracy.append(accuracy)
        bootstrap_loss.append(loss)
    mean_acc = np.mean(bootstrap_acc)
    mean_loss = np.mean(bootstrap_loss)
    return mean_acc, mean_loss
#bootstrap_acc,bootstrap_loss = boostrap_binary_cross_entropy(train_data,train_labels,1000)
#print(f"Bootstrap Test Accuracy: {test_acc}, Bootstrap Test Loss: {test_loss}")
plt.figure(figsize=(11,8))
counter =   0
for dimensions,history  in  emb_history.items():
    #plt.plot(history.history['acc'])
    #plt.plot(history.history['loss'])
    # Plot training accuracy
#    plt.plot(history['accuracy'], label=f'Train Acc (Embedding Dim {dimensions})', linestyle='-')
    # Plot validation accuracy
#    plt.plot(history['val_accuracy'], label=f'Val Acc (Embedding Dim {dimensions})', linestyle='--')
    plt.plot(history['loss'], label=f'Train Loss Acc (Embedding Dim {dimensions})', linestyle='-',color=colors[counter])
    plt.plot(history['val_loss'], label=f'Val Loss (Embedding Dim {dimensions})', linestyle='-.',color=colors[counter])
    counter +=  1
plt.title('Training and Validation Accuracy and Loss over Epochs for Different Embedding Dimensions')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')  # Automatically create a legend with appropriate labels
plt.grid(True)
plt.show()
model.save('my_model.keras')
