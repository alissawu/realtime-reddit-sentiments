import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchtext.data import imdb

#from keras.datasets import imdb
#ffrom keras.preprocessing.sequence import pad_sequences

# Load IMDB dataset
vocabulary_size = 30000
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


# Define the model
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

# Train the model
num_epochs = 6
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Validate the model
model.eval()
val_loss = 0
val_accuracy = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_accuracy += ((outputs > 0.5) == labels).float().mean().item()
val_loss /= len(val_loader)
val_accuracy /= len(val_loader)
print('Validation Loss:', val_loss)
print('Validation Accuracy:', val_accuracy)

# Evaluate the model on the test set
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
test_accuracy = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs).squeeze()
        test_accuracy += ((outputs > 0.5) == labels).float().mean().item()
test_accuracy /= len(test_loader)
print('Test Accuracy:', test_accuracy)

# Save the model
torch.save(model.state_dict(), 'sentiment_model.pth')