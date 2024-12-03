import torch
import torch.nn as nn
import torch.nn.functional as F
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

from epoch_test import batch_size, train_loader

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

pretrained_vectors = TEXT.vocab.vectors

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
    def __init__(self,vocab_size, embedding_dim, pretrained_vecs,batch_size):
        super(cnnToLSTMCustom,self).__init__()
        #top k2 k4
        #range(0,256,1)
        self.embed  =   nn.Embedding(vocab_size, embedding_dim)
        self.embed.weight.data.copy_(pretrained_vecs)
        self.embed.weight.requires_grad = False
        self.batch_size = batch_size

        self.kern2s1 =   nn.Conv1d(in_channels=256,out_channels=300,kernel_size=2,stride=1) #255
        self.kern4s2 = nn.Conv1d(in_channels=256, out_channels=300, kernel_size=4, stride=2)#127
        #mid k3 k6
        self.kern3s3p1 =   nn.Conv1d(in_channels=256,out_channels=300,kernel_size=3,stride=3, padding=1)#(253+2*1)/3+1=86
        self.kern6s3p1 =   nn.Conv1d(in_channels=256,out_channels=300,kernel_size=6,stride=3, padding=1)#(250+2*1)/3+1 = 85
        #bottom k4
        self.kern5s3 =   nn.Conv1d(in_channels=256,out_channels=300,kernel_size=5,stride=3,padding=2)#(251+2*2)/3+1=86
        self.uppLSTM    =   nn.LSTM(300, 512, batch_first=True,bidirectional=True)
        self.midLSTM    =   nn.LSTM(300, 512, batch_first=True,bidirectional=True)
        self.lowLSTM    =   nn.LSTM(300, 512, batch_first=True,bidirectional=True)

        self.weights    =   nn.Parameter(nn.torch([],dtype=torch.float))

        self.fc1    =   nn.Linear(256,16)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(16,2)

    def forward(self,x):
        x   =   self.embed(x).permute(0,2,1)
        embedding_tensor    =   torch.zeros(self.batch_size, embedding_dim, 512)

        embedding_tensor[:, :, 1::2]    =   x
        topk2   =   self.kern2s1(x)
        transform_topk2 =   self.kern2ImagTransformer(topk2.transpose(1,2))
        topk4   =   self.kern4s2(x)
        transform_topk4 =   self.kern4ImagTransformer(topk4.transpose(1,2))
        #upper   =   torch.cat([transform_topk2,transform_topk4],dim=-1)
        upper   =   transform_topk2 +   transform_topk4
        midk3=self.kern3s3p1(x)
        transform_midk3 = self.kern3ImagTransformer(midk3.transpose(1,2))
        midk6=self.kern6s3p1(x)
        transform_midk6 = self.kern6ImagTransformer(midk6.transpose(1,2))
        middle  =   transform_midk3 +   transform_midk6
        lowk5 = self.kern5s3p1(x)
        transform_lowk5 = self.kern5ImagTransformer(lowk5.transpose(1,2))
        lower  =    embedding_tensor    +   transform_lowk5
        upp_outputs,_ =   self.uppLSTM(upper)
        mid_outputs,_ =   self.midLSTM(middle)
        low_outputs,_ =   self.lowLSTM(lower)

        pair12  =   upp_outputs +   mid_outputs
        pair23  =   mid_outputs +   upp_outputs
        pair13  =   low_outputs +   upp_outputs
        trip    =   upp_outputs +   mid_outputs +   low_outputs

        normedWeights   =   F.softmax(self.weights,dim=0)



        fused   =   torch.mean(normedWeights[0]    *   pair12,
            normedWeights[1]    *   pair23,
            normedWeights[2]    *   pair13,
            normedWeights[3]    *   trip, dim=1)

        even_cells = fused[:, 0::2, :]  # Select even indices
        odd_cells = fused[:, 1::2, :]
        crunched = torch.cat(even_cells, odd_cells, dim=-1)
        swisher =   nn.SiLU(self.fc1(crunched))
        dropOuts    =   self.dropout(swisher)
        outputs =   F.softmax(self.fc2(dropOuts),dim=1)
        return outputs





    def kern2ImagTransformer(self,input_tensor):
        # Original tensor of shape (N, 300, 255)
        N, seq_len, num_filters = 4, 300, 255  # Example sizes
        output_tensor=input_tensor.to(dtype=torch.complex64)
        #input_tensor = torch.randn(N, seq_len, num_filters)  # Random data
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

    def kern4ImagTransformer(self,input_tensor):
        # Original tensor of shape (N, 300, 127)
        N, embedding_dim, num_filters = 4, 300, 127  # Example sizes
        input_tensor=input_tensor.to(dtype=torch.complex64)
        output_tensor = torch.zeros(N, embedding_dim, 256 * 2,dtype=torch.complex64)
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
        N, embedding_dim, num_filters = 16, 300, 86  # Example sizes
        input_tensor=input_tensor.to(dtype=torch.complex64)
        output_tensor = torch.zeros(N, embedding_dim, 256 * 2,dtype=torch.complex64)

        #values for the outlier 0 filter
        output_tensor[:, :, [1, 3]] = input_tensor[:, :, 0].unsqueeze(-1)
        for i in range(1, 85):
            indices = [6*i-1, 6*i+1, 6*i+3]
            output_tensor[:, :, indices] = input_tensor[:, :, i].unsqueeze(-1)

        #values for the outlier 85 filter
        output_tensor[:, :, [509, 511]] = input_tensor[:, :, 85].unsqueeze(-1)

    def kern6ImagTransformer(self,input_tensor):

        # Original tensor of shape (N, 300, 85)
        N, embedding_dim, num_filters = 16, 300, 85  # Example sizes
        input_tensor=input_tensor.to(dtype=torch.complex64)
        output_tensor = torch.zeros(N, embedding_dim, 256 * 2, dtype=torch.complex64)

        # Outlier filter 0
        output_tensor[:, :, [1, 3, 4, 6, 8]] = 1j * input_tensor[:, :, 0].unsqueeze(-1)  # Make values imaginary

        # Regular filters 1 to 83
        for i in range(1, 84):
            indices = [6*i-1, 6*i+1, 6*i+3, 6*i+4, 6*(i+1), 6*(i+1)+2]
            output_tensor[:, :, indices] = 1j * input_tensor[:, :, i].unsqueeze(-1).repeat(1, 1, 6)  # Make values imaginary

        # Outlier filter 84
        output_tensor[:, :, [503, 505, 507, 508, 510]] = 1j * input_tensor[:, :, 84].unsqueeze(-1)  # Make values imaginary

    def kern5ImagTransformer(self,input_tensor):
        """
        Transform input tensor of shape (N, 300, 86) into (N, 300, 512)
        with specified index mapping, making all assigned values imaginary.
        """
        # Get dimensions of the input tensor
        N, seq_len, num_filters = input_tensor.shape

        # Step 1: Create an output tensor of zeros with shape (N, 300, 512), as complex type
        output_tensor = torch.zeros(N, seq_len, 512, dtype=torch.complex64)

        # Step 2: Assign imaginary values for outlier filter 0
        output_tensor[:, :, [1, 3, 5]] = 1j * input_tensor[:, :, 0].unsqueeze(-1)

        # Step 3: Assign imaginary values for regular filters 1 to 84
        for i in range(1, 85):
            indices = [
                6 * (i - 1) + 2,
                6 * (i - 1) + 4,
                6 * (i - 1) + 7,
                6 * (i - 1) + 9,
                6 * (i - 1) + 11
            ]
            output_tensor[:, :, indices] = 1j * input_tensor[:, :, i].unsqueeze(-1)

        # Step 4: Assign imaginary values for outlier filter 85
        output_tensor[:, :, [506, 508, 511]] = 1j * input_tensor[:, :, 85].unsqueeze(-1)

        return output_tensor




"""class   initialSentModel(nn.Module):
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
"""

(X_train, y_train), (X_test, y_test) = IMDB.load_data(num_words=vocab_size)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

model = cnnToLSTMCustom(vocab_size,300,pretrained_vectors,batch_size)#SentimentAnalysisModel(vocabulary_size, embedding_size, lstm_size, max_words)


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