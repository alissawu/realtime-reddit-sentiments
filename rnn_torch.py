import torch
import torch.nn as nn
import torch.optim as optim
from attr.validators import max_len
from jsonschema.benchmarks.contains import middle
#from torch.utils.tensorboard    import  SummaryWriter
from torchtext.datasets import IMDB
from torchtext.data import Field,   LabelField, BucketIterator, DataLoader, TensorDataset, random_split
from torchtext.vocab    import vocab,   GloVe
#from    collections import  Counter,    OrderedDict
#https://saifgazali.medium.com/n-gram-cnn-model-for-sentimental-analysis-bb2aadd5dcb0

import numpy    as np
import requests

from epoch_test import batch_size

#params
max_len =   256
padding_type    =   'post'
vocab_size  =   65536
embedding_dim   =   300

#hypers
batch_size  =   16
epoch_count =   15
lr      =   0.004
min_lr  =   0.0005

TEXT    =   Field(sequential=True,  tokenize='spacy', lower=True,   batch_first=True,   fix_length=max_len, include_lengths=True)
LABEL   =   LabelField(dtype=torch.float, batch_first=True)
train_data, test_data = IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=embedding_dim))
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
class   cnnToLSTMCustom(nn.Module):
    def __init__(self):
        super(cnnToLSTMCustom,self).__init__()
        #top k2 k4
        #range(0,256,1)
        self.kern2s1 =   nn.Conv1d(in_channels=256,out_channels=300,kernel_size=2,stride=1) #255
        self.kern4s2 = nn.Conv1d(in_channels=256, out_channels=300, kernel_size=4, stride=2)#127
        #mid k3 k6
        self.kern3s3p1 =   nn.Conv1d(in_channels=256,out_channels=300,kernel_size=3,stride=3, padding=1)#(253+2*1)/3+1=86
        self.kern6s3p1 =   nn.Conv1d(in_channels=256,out_channels=300,kernel_size=6,stride=3, padding=1)#(250+2*1)/3+1 = 85
        #bottom k4
        self.kern5s3 =   nn.Conv1d(in_channels=256,out_channels=300,kernel_size=5,stride=3,padding=1)#(251+2*2)/3+1=86
    def forward(self,x_inp):
        topk2   =   self.kern2s1(x_inp)
        transform_topk2 =   self.kern2ImagTransformer(topk2.transpose(1,2))
        topk4   =   self.kern4s2(x_inp)
        transform_topk4 =   self.kern4ImagTransformer(topk4.transpose(1,2))
        upper   =   torch.cat([transform_topk2,transform_topk4],dim=-1)
        midk3=self.kern3s3p1(x_inp)
        transform_midk3 = self.kern3ImagTransformer(midk3.transpose(1,2))
        midk6=self.kern6s3p1(x_inp)
        transform_midk6 = self.kern6ImagTransformer(midk6.transpose(1,2))
        middle  =   torch.cat([transform_midk3,transform_midk6],dim=-1)
    def kern2ImagTransformer(self,input_tensor):
        # Original tensor of shape (N, 300, 255)
        N, seq_len, num_filters = 4, 300, 255  # Example sizes
        input_tensor = torch.randn(N, seq_len, num_filters)  # Random data

        ## Create index mapping for placement
        #indices = torch.arange(255).unsqueeze(0) * 2 + 1  # Calculate (2i+1)
        #indices = indices.repeat(N, seq_len, 1)  # Repeat for batch and sequence

        # Create the output tensor
        #output_tensor = torch.zeros(N, seq_len, 512)

        # Assign values
        #output_tensor[:, :, 1:-1:2] = input_tensor  # Populate indices (2i+1)
        #output_tensor[:, :, 2:-1:2] = input_tensor  # Populate indices (2i+2)
        #
        #
        #
        # Step 2: Assign values for each filter to the mapped indices
        for i in range(num_filters):
            # For each filter, map to positions 2i+1 and 2i+2
            output_tensor[:, :, 2*i+1] = input_tensor[:, :, i]
            output_tensor[:, :, 2*i+2] = input_tensor[:, :, i]

        print(output_tensor.shape)  # Should be (N, 300, 512)



    def kern4ImagTransformer(self,output_tensor):
        # Original tensor of shape (N, 300, 127)
        N, seq_len, num_filters = 4, 300, 127  # Example sizes


        # Step 1: Create an output tensor of zeros with shape (N, 300, 512), as complex type
        output_tensor = torch.zeros(N, seq_len, 256 * 2, dtype=torch.complex64)

        # Step 2: Assign imaginary values for each filter
        for i in range(num_filters):
            # Compute target indices for filter i
            indices = [4*i+1, 4*i+3, 4*i+4, 4*i+6]
            # Assign the input filter values as imaginary numbers to the output at the computed indices
            output_tensor[:, :, indices] = 1j * input_tensor[:, :, i].unsqueeze(-1)
        """# Step 3: Populate the indices for each filter, making values imaginary
        for idx, i in enumerate(range(0, len(indices), 4)):
            # Assign the input values to the imaginary part of the output tensor
            output_tensor[:, :, indices[i:i+4]] = 1j * input_tensor[:, :, idx].unsqueeze(-1).repeat(1, 1, 4)
        """

    def kern3Transformer(self,input_tensor):
        # Original tensor of shape (N, 300, 86)
        N, seq_len, num_filters = 4, 300, 86  # Example sizes
        input_tensor = torch.randn(N, seq_len, num_filters)  # Random data

        # Step 1: Create an output tensor of zeros with shape (N, 300, 512)
        output_tensor = torch.zeros(N, seq_len, 256 * 2)

        # Step 2: Assign values for the outlier 0 filter
        output_tensor[:, :, [1, 3]] = input_tensor[:, :, 0].unsqueeze(-1)

        # Step 3: Assign values for regular filters 1 to 84
        for i in range(1, 85):
            indices = [6*i-1, 6*i+1, 6*i+3]
            output_tensor[:, :, indices] = input_tensor[:, :, i].unsqueeze(-1)

        # Step 4: Assign values for the outlier 85 filter
        output_tensor[:, :, [509, 511]] = input_tensor[:, :, 85].unsqueeze(-1)

    #Kern6 with imag

    def kern6ImagTransformer(self,input_tensor):

        # Original tensor of shape (N, 300, 85)
        N, seq_len, num_filters = 4, 300, 85  # Example sizes
        input_tensor = torch.randn(N, seq_len, num_filters)  # Random data

        # Step 1: Create an output tensor of zeros with shape (N, 300, 512), as complex type
        output_tensor = torch.zeros(N, seq_len, 256 * 2, dtype=torch.complex64)

        # Step 2: Populate the indices for each filter
        # Outlier filter 0
        output_tensor[:, :, [1, 3, 4, 6, 8]] = 1j * input_tensor[:, :, 0].unsqueeze(-1)  # Make values imaginary

        # Regular filters 1 to 83
        for i in range(1, 84):
            indices = [6*i-1, 6*i+1, 6*i+3, 6*i+4, 6*(i+1), 6*(i+1)+2]
            output_tensor[:, :, indices] = 1j * input_tensor[:, :, i].unsqueeze(-1).repeat(1, 1, 6)  # Make values imaginary

        # Outlier filter 84
        output_tensor[:, :, [503, 505, 507, 508, 510]] = 1j * input_tensor[:, :, 84].unsqueeze(-1)  # Make values imaginary

        # Step 3: Validate the result
        print(output_tensor.shape)  # Should be (N, 300, 512)



















class   initialSentModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_units,pre_train_embeds):
        super(initialSentModel,    self).__init__()
        self.recurrDropout =   0.25
        self.embedding = nn.Embedding.from_embedding(pre_train_embeds,freeze=False)
                #max_norm (float, optional) – See module initialization documentation.
                #norm_type (float, optional) – See module initialization documentation. Default 2.
                #scale_grad_by_freq (bool, optional) – See module initialization documentation. Default False.
                #sparse (bool, optional) – See module initialization documentation.
        self.lstm1 = nn.LSTM(300, 512, batch_first=True,bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True, bidirectional=True)


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