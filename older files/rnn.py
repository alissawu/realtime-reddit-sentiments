# what i was doing real quick before i realized you commited a torch thing LMNFAO sigh
#from ensurepip import bootstrap
import requests
import tensorflow as tf
# noinspection PyUnresolvedReferences
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, GlobalAveragePooling1D, Dense, Dropout
from keras.datasets import imdb
from keras.src.layers import GlobalMaxPooling1D
from tensorflow.keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import swish
from tensorflow.keras.regularizers import l2

from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
#from keras.src.layers import TextVectorization
# const for data preprocessing
max_length = 256  #2^8 max length of the sequences
padding_type = 'post'  # padding type for sequences shorter than the maximum length
vocab_size = 65536  # size of the vocabulary used in the Embedding layer
#5000,10000, 20000
# Load the IMDB dataset

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# Get word index mapping integer IDs to words
word_index = imdb.get_word_index()

# word index offset, so shift it for compatibility with our dataset
word_index = {k: (v + 3) for k, v in word_index.items()}  # Adjust for special tokens
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# reverse dictionary to map integer IDs to words
reverse_word_index = {value: key for (key, value) in word_index.items()}

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


# get full vocab
url =   "http://nlp.stanford.edu/data/glove.6B.100d.txt"
top_words = {word: index for word, index in word_index.items() if index < vocab_size}
reverse_word_index = {index: word for word, index in top_words.items()}
embedding_index = {}
#stream glove embeddings for  top IMDb vocabulary, store only
response = requests.get(url, stream=True)
for line in response.iter_lines():
    if line:
        decoded_line = line.decode('utf-8')
        values = decoded_line.split()
        word = values[0]
        if word in top_words:  # Only store embeddings if the word is in our IMDb vocabulary
            coefficients = list(map(float, values[1:]))
            embedding_index[word] = coefficients


#embedding_dimensions    =   [2**i for i in range(4,9)]
embedding_dimension    =   100

embedding_matrix = np.zeros([vocab_size, embedding_dimension])

for word,i in word_index.items():
    if  i   < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
# Model architecture
# embedding_dim from 16 to 256
# i
def build_model(vocab_size, embedding_dim=256, hidden_units=16, embedding_matrix=None):
    model = Sequential([
        #inputs 256 x 1
        #inputs
        Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True, weights=[embedding_matrix]),
        # outputs 256 x 1 x embedding_dim
        Bidirectional(LSTM(256, activation='tanh', return_sequences=True, dropout=0.25, recurrent_dropout=0.25)),
        Bidirectional(LSTM(128, activation='tanh', return_sequences=True, dropout=0.25, recurrent_dropout=0.25)),
        Dense(64, activation    =   swish),
        GlobalMaxPooling1D(),
        #GlobalAveragePooling1D(),
        #outputs 256  x embedding_data
        Dropout(0.25),
        # outputs 256 x embedding_data
        Dense(hidden_units, activation='relu', kernel_regularizer=l2(0.04)),
        Dense(3, activation='softmax')


        #Dense(1, activation='sigmoid')
    ])
    return model
def build_model_initial(vocab_size, embedding_dim=64, hidden_units=16):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dropout(0.25),
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
#ideal is 16 emb dim + epoch = 4

colors = ['red', 'orange', 'green', 'blue', 'purple']
#colors = ['red', 'green', 'blue', 'orange', 'purple']
emb_history =   {}
"""for i in embedding_dimensions:
    print(f"\nModel with embedding dimension {i}: ")

    model1 = build_model(vocab_size,i)
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model1.summary()

# Train and evaluate our model based on imdb review data ONLY - no reddit yet

    
    
     = model1.fit(train_data, train_labels, epochs=8, batch_size=16, validation_data=(val_data, val_labels), verbose=2)
    emb_history[i] = history.history
    test_loss, test_acc = model1.evaluate(test_data, test_labels, verbose=2)"""
class neurActivationMonitor(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
    def on_epoch_end(self,epoch,logs=None):
        for layer_indexer,layer in enumerate(self.model.layers):
             if 'dense' in  layer.name  or 'lstm'   in  layer.name:
                weights    =   layer.get_weights()
                if  weights:
                    activations =   self.model.predict(self.validation_data   )
                    zero_activations    =   np.sum(activations==0)
                    total_neurons   =   activations.size
                    zero_percent  =   100 * zero_activations / total_neurons
                    print(f'EPOCH {epoch}, LAYER {layer.name}  -   {zero_percent}% of  neurons are dead')
monitor = neurActivationMonitor(validation_data=val_data)

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

model1 = build_model(vocab_size,embedding_dimension, 16, embedding_matrix)
# model1.compile(optimizer=Adam(learning_rate=0.004), loss='binary_crossentropy', metrics=['accuracy'])
model1.compile(optimizer=Adam(learning_rate=0.004), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.summary()

# Train and evaluate our model based on imdb review data ONLY - no reddit yet
reduce_lr   =   ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience   =   2,  min_lr  =   0.00025)
early_stopper   =   EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model1.fit(train_data, train_labels, epochs=15, batch_size=16, validation_data=(val_data, val_labels), callbacks=[tensorboard_callback, monitor, reduce_lr, early_stopper])
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
    mean_acc = np.mean(bootstrap_accuracy)
    mean_loss = np.mean(bootstrap_loss)
    return mean_acc, mean_loss
#bootstrap_acc,bootstrap_loss = boostrap_binary_cross_entropy(train_data,train_labels,1000)
#print(f"Bootstrap Test Accuracy: {test_acc}, Bootstrap Test Loss: {test_loss}")
plt.figure(figsize=(14,10))
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


num_bootstrap_samples = 400  # Adjust this as needed for more or fewer bootstrap samples
epochs = 5
bootstrap_accuracies = []

for i in range(num_bootstrap_samples):
    print(f"Bootstrap Sample {i + 1}/{num_bootstrap_samples}")

    train_data_resampled, train_labels_resampled = resample(train_data, train_labels, replace=True)

    model = build_model(vocab_size, embedding_dim=256, hidden_units=16, embedding_matrix=embedding_matrix)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data_resampled, train_labels_resampled, epochs=epochs, validation_data=(test_data, test_labels),
              verbose=0)
    #pop an argmax on
    predictions = np.argmax(model.predict(test_data),axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    accuracy = accuracy_score(true_classes, predicted_classes)
    bootstrap_accuracies.append(accuracy)
    print(f"Accuracy for sample {i + 1}: {accuracy:.4f}")

mean_accuracy = np.mean(bootstrap_accuracies)
std_dev_accuracy = np.std(bootstrap_accuracies)
conf_interval = (mean_accuracy - 1.96 * std_dev_accuracy, mean_accuracy + 1.96 * std_dev_accuracy)

print(f"\nBootstrap Results:")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"95% Confidence Interval for Accuracy: {conf_interval}")

plt.title('Training and Validation Accuracy and Loss over Epochs for Different Embedding Dimensions')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')  # Automatically create a legend with appropriate labels
plt.grid(True)
plt.show()
model.save('my_model.keras')