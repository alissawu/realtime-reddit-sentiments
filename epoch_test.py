import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load IMDB dataset
vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

# Pad sequences to ensure uniform length
max_words = 500
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
batch_size = 64
dataset = TensorDataset(X_train, y_train)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Should be the same as the model in rnn_torch.py
class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, max_words):
        super(SentimentAnalysisModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, lstm_size, batch_first=True)
        self.fc = nn.Linear(lstm_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        x = self.sigmoid(x)
        return x

embedding_size = 32
lstm_size = 100
model = SentimentAnalysisModel(vocabulary_size, embedding_size, lstm_size, max_words)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Store training and validation accuracy for plotting
train_accuracies = []
val_accuracies = []

# Train the model and capture accuracy at each epoch
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_train = (outputs > 0.5).float()
        correct_train += (predicted_train == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Validate the model after each epoch
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze()
            predicted_val = (outputs > 0.5).float()
            correct_val += (predicted_val == labels).sum().item()
            total_val += labels.size(0)
    val_accuracy = correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Plot accuracy vs. epochs
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')
plt.legend()
plt.show()


