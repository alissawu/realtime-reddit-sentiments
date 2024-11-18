import torch
import torch.nn as nn
import torch.optim as optim
from attr.validators import max_len
#from torch.utils.tensorboard    import  SummaryWriter
from torchtext.datasets import IMDB
from torchtext.data import Field,   LabelField, BucketIterator, DataLoader, TensorDataset, random_split
from torchtext.vocab    import vocab,   GloVe
#from    collections import  Counter,    OrderedDict
import numpy    as np
import requests

from epoch_test import batch_size

#params
max_len =   256
padding_type    =   'post'
vocab_size  =   65536
embedding_dim   =   100

#hypers
batch_size  =   16

TEXT    =   Field(sequential=True,  tokenize='spacy', lower=True,   batch_first=True,   fix_length=max_len, include_lengths=True)
LABEL   =   LabelField(dtype=torch.float, batch_first=True)
train_data, test_data = IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=100))
LABEL.build_vocab(train_data)

split_1 = 5 / 2
split_2 = 20 / 17
split_1_index = int(len(test_data) // split_1)
split_2_index = int(len(test_data) // split_2) + 1

train_data.examples += test_data.examples[:split_1_index]
val_data = test_data.examples[split_1_index:split_2_index]
test_data = test_data.examples[split_2_index:]

from torchtext.data.dataset import  Dataset
val_data    =   Dataset(val_data,fields={'text':TEXT,'label':LABEL})
test_data   =   Dataset(test_data,fields={'text':TEXT,'label':LABEL})



train_iter, test_iter = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
)


# Load IMDB dataset
train_data,test_data    =   IMDB()
train_labels,   test_labels =   [], []
train_seq,  test_seq    =   []

word_counter    =   Counter()
for label,  line    in  train_data:
    train_seq.append(line)
    train_labels.append(1   if  label   ==  'pos'   else    0)

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