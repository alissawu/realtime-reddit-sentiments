import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
#from dask.dataframe import test_dataframe
#from more_itertools.more import padded
#from attr.validators import max_len
#from jsonschema.benchmarks.contains import middle
#from torch.utils.tensorboard    import  SummaryWriter
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab    import Vocab, build_vocab_from_iterator,   GloVe
from torch.utils.data import ConcatDataset



from    collections import  Counter,    OrderedDict
#https://saifgazali.medium.com/n-gram-cnn-model-for-sentimental-analysis-bb2aadd5dcb0

import numpy    as np
import requests

#from epoch_test import batch_size, train_loader


class   cnnToLSTMCustom(nn.Module):
    def __init__(self,vocab_size:   int , embedding_dim:    int , pretrained_vecs ,batch_size:    int ):
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

        self.weights    =   nn.Parameter(torch.tensor([0.25,0.25,0.25,0.25],dtype=torch.float))

        self.fc1    =   nn.Linear(256,16)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(16,2)
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)

        # CNN Layers
        topk2 = self.kern2s1(x)
        topk4 = self.kern4s2(x)
        midk3 = self.kern3s3p1(x)
        midk6 = self.kern6s3p1(x)
        lowk5 = self.kern5s3(x)

        # LSTM Outputs
        upp_outputs, _ = self.uppLSTM(topk2.transpose(1, 2) + topk4.transpose(1, 2))
        mid_outputs, _ = self.midLSTM(midk3.transpose(1, 2) + midk6.transpose(1, 2))
        low_outputs, _ = self.lowLSTM(lowk5.transpose(1, 2))

        # Apply PLA
        def apply_pla(features):
            # Compute covariance matrix
            cov_matrix = features.T @ features
            eigvals, eigvecs = torch.linalg.eigh(cov_matrix)

            # Sort eigenvectors by eigenvalues in descending order
            sorted_indices = torch.argsort(eigvals, descending=True)
            top_k_eigvecs = eigvecs[:, sorted_indices[:self.num_components]]

            # Project features onto top principal components
            return features @ top_k_eigvecs

        upp_features = apply_pla(upp_outputs.mean(dim=1))
        mid_features = apply_pla(mid_outputs.mean(dim=1))
        low_features = apply_pla(low_outputs.mean(dim=1))

        # Combine PLA-reduced features (simple concatenation or addition)
        fused = upp_features + mid_features + low_features  # Replace learned weights with direct combination

        # Fully Connected Layers
        swisher = F.silu(self.fc1(fused.mean(dim=1)))
        dropout = self.dropout(swisher)
        outputs = F.softmax(self.fc2(dropout), dim=1)

        return outputs

        # noinspection PyUnreachableCode
        print("""    def forward(self,x):
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
        crunched = torch.cat((even_cells, odd_cells), dim=-1)
        swisher =   nn.SiLU(self.fc1(crunched))
        dropOuts    =   self.dropout(swisher)
        outputs =   F.softmax(self.fc2(dropOuts),dim=1)
        return outputs""")




    def kern2ImagTransformer(input_tensor):
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
        return output_tensor

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
        return output_tensor
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
        return output_tensor
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
        return output_tensor
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



#cutesry
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

"""
def preprocess_data(data_iter, vocab, max_tokens):
    preprocessed = []
    padding_idx = 0
    print(dir(data_iter))
    print(type(data_iter))
    for label, text in data_iter:
        print(label)
        tokenized_text = token_retriever(text)
        token_ids = vocab(tokenized_text)[:max_len]
        padding_needed = max_tokens - len(token_ids)
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding  # Handle odd-length padding

        padded_text = [padding_idx] * left_padding + token_ids + [padding_idx] * right_padding
        preprocessed.append((torch.tensor(padded_text, dtype=torch.long),
                             torch.tensor(1.0 if label == "pos" else 0.0, dtype=torch.float)))"""


def process_dataset(combined_dataset=Dataset):
    #comp sizes for train and initial test splits
    total_size = 50000
    train_size = int(total_size * 0.7)
    val_size    = int(total_size * 0.2)
    test_size = int(total_size * 0.1)
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for label, text in data_iter:
            yield tokenizer(text)

    def text_pipeline(text):
        return tokenizer(text)

        #return torch.tensor(tokenizer(text), dtype=torch.int64)

    def label_pipeline(label):
        if isinstance(label, str):
            if label == "pos":
                return torch.tensor(1, dtype=torch.float)
            elif label == "neg":
                return torch.tensor(0, dtype=torch.float)
            else:
                raise ValueError(f"Unexpected label: {label}")
        elif isinstance(label, int) or label.isdigit():
            return torch.tensor(float(int(label) - 1), dtype=torch.float)
        else:
            raise ValueError(f"Unsupported label type: {label}")
    def pipeline_driver(raw_data_split):
        print(f"   {max([len(text_pipeline(text)) for _, text in raw_data_split])}")
        return  [(label_pipeline(label),text_pipeline(text))
                 for label, text in raw_data_split
        ]

    # Create train and test datasets with random split
    train_dSet, test_dSet,val_dSet = random_split(combined_dataset, [train_size,val_size, test_size])
    #train_size =   25000+25000//split_1 or 50000*0.7
    #test_size   =   25000(1-1//split_1) or 50000*0.1
    #val_size   =   25000(8/10-2/5) or 50000*0.2

    # Preprocess all datasets using label and text pipelines
    return (train_dSet, val_dSet, test_dSet), (pipeline_driver(train_dSet),pipeline_driver(val_dSet),pipeline_driver(test_dSet))



# params
max_len = 256
padding_type = 'post'
vocab_size = 65536
embedding_dim = 300

# hypers
batch_size = 16
epoch_count = 15
learning_rate = 0.004
min_lr = 0.0005

token_retriever = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for _, text in data_iter:
        if isinstance(text, str):  # If `text` is raw text
            yield token_retriever(text)
        elif isinstance(text, list):  # If `text` is already tokenized
            yield text  # Use it directly without tokenizing again
        else:
            raise ValueError("Unexpected text format. Expected string or list of tokens.")


# Custom iterwrapper
class redoneTupleDataset(Dataset):
    def __init__(self, data):
        self.data = list(data)  # Convert iterable to a list for indexing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        return label, text


# Convert the datasets into PyTorch Dataset objects
iter1 = IMDB(root=".data", split='train')
iter2 = IMDB(root=".data", split='test')
iter1_wrapped = redoneTupleDataset(iter1)
iter2_wrapped = redoneTupleDataset(iter2)

glove = GloVe(name="6B", dim=embedding_dim)
glove_path = os.path.expanduser(
    "C:\\Users\\epw268\\Documents\\GitHub\\realtime-reddit-sentiments\\.vector_cache\\glove.6B.300d.txt")  # Adjust for your cache path
GloVe_itos = []

# Combine them into one dataset https://discuss.pytorch.org/t/how-does-concatdataset-work/60083
combined_dataset = ConcatDataset([iter1_wrapped, iter2_wrapped])
(train_dSet, val_dSet, test_dSet), (train_data, val_data, test_data) = process_dataset(combined_dataset)
vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>", "<pad>"])
vocab_size = int(len(vocab.get_stoi()) // 2 + 1)
vocab.set_default_index(vocab["<unk>"])


glove_path = os.path.expanduser(
    "C:\\Users\\epw268\\Documents\\GitHub\\realtime-reddit-sentiments\\.vector_cache\\glove.6B.300d.txt")  # Adjust for your cache path
GloVe_itos = []

""""# Read the GloVe file to extract tokens
with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        token = line.split()[0]  # First element is the token
        GloVe_itos.append(token)
#GloVe_itos  =   GloVe.Vocab.get_itos()
"""
# stoi = {word: idx for idx, word in enumerate(GloVe_itos)}  # String-to-index mapping


# Simulate a vocabulary of size `vocab_size`
# Assuming the vocabulary is sorted by frequency (common practice in NLP tasks)
# "<unk>" and "<pad>" are added for unknown tokens and padding
print(dir(glove))
pad_idx = vocab["<pad>"]
# vocab_list = ["<pad>", "<unk>"] + list(stoi.keys())[:vocab_size - 2]

# Create vocab-to-index mapping
# word_to_index = {word: idx for idx, word in enumerate(vocab_list)}
# absurdly big auauauaua 100000000
pretrained_vectors = torch.zeros((10000000, embedding_dim))
# fix the vocab and use glove pretraining
for word, idx in vocab.get_stoi().items():
    if word in glove.stoi:  # Check if word is in GloVe's vocabulary
        # pretrained_vectors[idx] = stoi[word]

        pretrained_vectors[idx] = torch.tensor(glove.stoi[word], dtype=torch.float)
    elif word == "<pad>":  # Padding vector (optional, all zeros by default)
        pretrained_vectors[idx] = torch.zeros(embedding_dim)
    else:  # For OOV words (e.g., "<unk>")
        pretrained_vectors[idx] = torch.rand(embedding_dim)  # Random initialization

# Create PyTorch Embedding Layer
embedding_layer = torch.nn.Embedding.from_pretrained(pretrained_vectors, freeze=False)  # freeze=False to fine-tune

"""
def collate_batch(batch):
    # after separately pipelining, zip
    labels, texts = zip(*batch)
    labels = torch.tensor(labels)
    tokenizer=get_tokenizer("basic_english")
    #texts = vocab(tokenizer(texts))
    tokenized_texts =   [token_retriever(text)  for text in texts]
    newer=[vocab(tokenizations)    for tokenizations in tokenized_texts]
    tensorized_numbertexts   =   torch.tensor(newer)
    print(tensorized_numbertexts)
    for i in texts:
        print(i)
        print(type(i))
    text_lengths = [len(text) for text in texts]

    texts = pad_sequence(vocab(token_retriever(texts)), batch_first=True, padding_value=pad_idx)
    return labels, texts, text_lengths"""
def collate_batch(batch):
    # Unpack the batch into labels and texts
    labels, texts = zip(*batch)
    max_length = 4096
    # Convert labels to tensors
    labels = torch.tensor(labels)

    # Tokenize texts
    tokenized_texts = [token_retriever(text) for text in texts]

    # Numericalize tokens

    numericalized_texts = [torch.tensor(vocab.lookup_indices(list(tokens))) for tokens in tokenized_texts]
    if len(numericalized_texts[0]) < max_length:
        numericalized_texts[0] = torch.cat(
            [numericalized_texts[0], torch.full((max_length - len(numericalized_texts[0]),), pad_idx)]
        )
    else:
        numericalized_texts[0] = numericalized_texts[0][:max_length]    # Pad sequences
    padded_texts = pad_sequence(numericalized_texts, batch_first=True, padding_value=pad_idx)

    # Calculate text lengths
    text_lengths = [len(tokens) for tokens in numericalized_texts]

    return labels, padded_texts#, text_lengths


dLoad_train = DataLoader(train_dSet, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_batch)
dLoad_val = DataLoader(val_dSet, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_batch)
dLoad_test = DataLoader(test_dSet, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_batch)
# inst_test = IMDB(split="test")
"""
class   IMDBDataset(Dataset):
    def __init__(self, dataset, tokenizer,vocab):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        label,text  =   self.dataset[idx]

        label_tensor = torch.tensor(1.0 if label == "pos" else 0.0, dtype=torch.float)
        text_tokens = self.tokenizer(text)
        text_tensor = torch.tensor([self.vocab[token] for token in text_tokens],dtype=torch.float)
        return  text_tensor, label_tensor
imdbDataset =   IMDBDataset(inst_train, token_retriever, stoi)


    text_list, label_list = [],[]
    for text, label in batch:
        text_list.append(text)
        label_list.append(label)
    text_padded =   pad_sequence(text_list, batch_first=True,   padding_value=stoi['<pad>'])
    labels  =   torch.tensor(label_list, dtype=torch.float)
    return text_padded, labels"""
# this one
all_texts = []
all_labels = []

for label_batch,    text_batch in dLoad_train:
    all_texts.append(text_batch)
    all_labels.append(label_batch)
# TRIAL PIECE
all_labels = [label[0] if isinstance(label, tuple) else label for label in all_labels]
if all(isinstance(label, torch.Tensor) for label in all_labels):
    train_labels_tensor = torch.cat(all_labels, dim=0)
else:
    raise TypeError("All elements in `all_labels` must be tensors.")
# END OF TRIAL PIECE
print(all_texts)
text_shapes =   [text.shape   for  text in all_texts]
print(text_shapes)
#dim_problems

train_texts_tensor = torch.cat(all_texts, dim=0)
print(train_texts_tensor)
train_labels_tensor = torch.cat(all_labels, dim=0)

"""import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

# Define tokenizer and vocabulary
tokenizer = get_tokenizer("basic_english")

# Function to yield tokens from the dataset
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# Load train split to build vocabulary
train_iter = IMDB(root=".data", split="train")
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Helper function to process text to tensor
def text_pipeline(text):
    #print(f"{len(vocab(tokenizer(text)))}   {tokenizer(text)}{text}    \n")
    #print(f"{max(len(tokenizer(text)))}")
    print(vocab(tokenizer(text)))
    return torch.tensor(vocab(tokenizer(text)), dtype=torch.int64)

# Helper function to process labels to tensor
def label_pipeline(label):
    if  isinstance(label, str):
        if label == "pos":
            return torch.tensor(1, dtype=torch.float)
        elif    label == "neg":
            return torch.tensor(0, dtype=torch.float)
        else:
            raise ValueError(f"Unexpected label: {label}")
    elif    isinstance(label, int)  or label.isdigit():
        return torch.tensor(int(label)-1, dtype=torch.float)
    else:
        raise ValueError(f"Unsupported label type: {label}")
    # Load train and test data with transformations
def process_dataset(split):
    raw_iter = IMDB(root=".data", split=split)
    data = [
        (label_pipeline(label), text_pipeline(text))
        for label, text in raw_iter
    ]
    print(f"{split}   {max([len(text_pipeline(text)) for _, text in raw_iter])}")
    return data

train_data = process_dataset("train")
test_data = process_dataset("test")

# Example: Pad sequences for batching
def collate_batch(batch):
    labels, texts = zip(*batch)
    labels = torch.stack(labels)
    text_lengths = [len(text) for text in texts]
    texts = pad_sequence(texts, batch_first=True)
    return labels, texts, text_lengths

# DataLoader for batching
from torch.utils.data import DataLoader

dLoad_train = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_batch)
dLoad_test = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_batch)



# Example: Iterate through the DataLoader
for labels, texts, lengths in dLoad_train:
    print("Labels:", labels)
    print("Texts:", texts)
    print("Lengths:", lengths)
    print(max(lengths))
    break
all_texts = []
all_labels = []

for text_batch, label_batch in dLoad_train:
    all_texts.append(text_batch)
    all_labels.append(label_batch)
# TRIAL PIECE
all_labels = [label[0] if isinstance(label, tuple) else label for label in all_labels]
if all(isinstance(label, torch.Tensor) for label in all_labels):
    train_labels_tensor = torch.cat(all_labels, dim=0)
else:
    raise TypeError("All elements in `all_labels` must be tensors.")
#END OF TRIAL PIECE
train_texts_tensor = torch.cat(all_texts,dim=0)
train_labels_tensor = torch.cat(all_labels,dim=0)


print(str(type(dLoad_train)) + ".trainer    |.    " + str(dir(dLoad_train)))
print(str(type(dLoad_test)) + ".    |.    " + str(dir(dLoad_test)))"""