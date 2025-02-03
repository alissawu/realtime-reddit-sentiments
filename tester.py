import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import    torch.utils.checkpoint as checkpoint
#from dask.dataframe import test_dataframe
#from more_itertools.more import padded
#from attr.validators import max_len
#from jsonschema.benchmarks.contains import middle
#from torch.utils.tensorboard    import  SummaryWriter
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab    import Vocab, build_vocab_from_iterator,   GloVe
from torch.utils.data import ConcatDataset

from tqdm import tqdm

from    collections import  Counter,    OrderedDict
#https://saifgazali.medium.com/n-gram-cnn-model-for-sentimental-analysis-bb2aadd5dcb0

import numpy    as np
import requests
from itertools  import tee
#from epoch_test import batch_size, train_loader


class   cnnToLSTMCustom(nn.Module):
    def __init__(self,vocab_size:   int , embedding_dim:    int , pretrained_vecs ,batch_size:    int , max_len:    int):
        super(cnnToLSTMCustom,self).__init__()
        #top k2 k4
        #range(0,256,1)
        self.max_len = max_len
        self.embed  =   nn.Embedding(vocab_size, embedding_dim)
        print(self.embed.weight.data.size())
        self.embed.weight.data.copy_(pretrained_vecs)
        self.embed.weight.requires_grad = False
        self.batch_size = batch_size
        self.num_components = 512

        self.kern2s1 =   nn.Conv1d(in_channels=300,out_channels=300,kernel_size=2,stride=1) #255 to 2047
        self.kern4s2 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=4, stride=2)#127 to 1023
        #mid k3 k6
        self.kern3s3p1 =   nn.Conv1d(in_channels=300,out_channels=300,kernel_size=3,stride=3, padding=2)#(2045+2*2)/3+1=684 from 86
        self.kern6s3p1 =   nn.Conv1d(in_channels=300,out_channels=300,kernel_size=6,stride=3, padding=2)#(2042+2*2)/3+1 = 683 from 85
        #bottom k4
        self.kern5s3 =   nn.Conv1d(in_channels=300,out_channels=300,kernel_size=5,stride=3,padding=0)#(2043)/3+1=682  from 84
        self.uppLSTM    =   nn.LSTM(300, 4096, batch_first=True,bidirectional=True, dtype=torch.complex64)
        self.midLSTM    =   nn.LSTM(300, 4096, batch_first=True,bidirectional=True, dtype=torch.complex64)
        self.lowLSTM    =   nn.LSTM(300, 4096, batch_first=True,bidirectional=True, dtype=torch.complex64)

        self.weights    =   nn.Parameter(torch.tensor([0.25,0.25,0.25,0.25],dtype=torch.float))

        self.fc1    =   nn.Linear(2048,16)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(32,2)
    def forward(self, x):
        print(f"X Input: {x}")
        x = self.embed(x)
        x   =        x.permute(0, 2, 1)

        # CNN Layers
        topk2   =   self.kern2ImagTransformer(self.kern2s1(x))
        topk4   =   self.kern4ImagTransformer(self.kern4s2(x))
        midk3   =   self.kern3ImagTransformer(self.kern3s3p1(x))
        midk6   =   self.kern6ImagTransformer(self.kern6s3p1(x))
        lowk5   =   self.kern5ImagTransformer(self.kern5s3(x))


        # 16,300,4096 into LSTM into Outputs
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





    def kern2ImagTransformer(self,  input_tensor):
        #batch_size,300,2047   k2s1
        N, seq_len, num_filters = self.batch_size, 300, 2047
        ## Create index mapping for placement
        #indices = torch.arange(255).unsqueeze(0) * 2 + 1  # Calculate (2i+1)
        #indices = indices.repeat(N, seq_len, 1)  # Repeat for batch and sequence

        # Create the output tensor
        output_tensor = torch.zeros(N, seq_len, 4096, dtype=torch.complex64)

        for i in range(num_filters):
            #[orig,overlap_back]
            # For each filter, map to positions 2i+1 and 2i+2
            output_tensor[:, :, 2*i+1] = input_tensor[:, :, i]
            output_tensor[:, :, 2*i+2] = input_tensor[:, :, i]

        return output_tensor
    def kern4ImagTransformer(self,input_tensor):
        #batch_size,300,1023  k4s2
        N, embedding_dim, num_filters = self.batch_size, 300, 1023  # Example sizes
        input_tensor=input_tensor.to(dtype=torch.complex64)
        output_tensor = torch.zeros(N, embedding_dim, 2048 * 2,dtype=torch.complex64)
        for i in range(num_filters):
            # Compute target indices for filter i
            indices = [4*i+1, 4*i+3, 4*i+4, 4*i+6]
            # Assign the input filter values as imaginary numbers to the output at the computed indices
            output_tensor[:, :, indices] = 1j * input_tensor[:, :, i].unsqueeze(-1)

        return output_tensor
    def kern3ImagTransformer(self,input_tensor):
        #batch_size,300,684     k3s3p2
        N, embedding_dim, num_filters = self.batch_size, 300, 684  # Example sizes
        input_tensor=input_tensor.to(dtype=torch.complex64)
        output_tensor = torch.zeros(N, embedding_dim, 2048 * 2,dtype=torch.complex64)
        #[][][,1]
        #[,3][,5][,7]
        #[][][]
        #[][][]
        # clip off the first filter at index 0
        #output_tensor[:, :, [1]] = input_tensor[:, :, 0].unsqueeze(-1)
        for i in range(1, 683):
            indices = [6*i-3, 6*i-1, 6*i+1]#4089,4091,4093
            output_tensor[:, :, indices] = input_tensor[:, :, i].unsqueeze(-1)

        #cut off outlier filter at index 683
#        output_tensor[:, :, [4095]] = input_tensor[:, :, 683].unsqueeze(-1)
        return output_tensor
    def kern6ImagTransformer(self,input_tensor):

        # Original tensor of shape (N, 300, 683)
        #batch_size,300,683 k6s3p2
        N, embedding_dim, num_filters = self.batch_size, 300, 683  # Example sizes
        input_tensor=input_tensor.to(dtype=torch.complex64)
        output_tensor = torch.zeros(N, embedding_dim, 2048 * 2, dtype=torch.complex64)
        # [][][,1][2,][4,][6,]
        # [,3][,5][,7][8,][10,][12,]
        # [][][][][][]
        # [][][][][][]
        # Outlier filter 0
        output_tensor[:, :, [1, 2, 4, 6]] = 1j * input_tensor[:, :, 0].unsqueeze(-1)  # Make values imaginary
        # Regular filters 1 to 682
        for i in range(1, 682):#3,5,7, 8 , 10, 12
            indices = [6*i-3, 6*i-1, 6*i+1, 6*i+2, 6*i+4,6*i+6]
            output_tensor[:, :, indices] = 1j * input_tensor[:, :, i].unsqueeze(-1).repeat(1, 1, 6)  # Make values imaginary
#12:58.  1/23/25
        # Outlier filter _4083, _4085, _4087, 4088_, 4090_, 4092_
        # Outlier filter _4089, _4091, _4093, 4094_, __, __

        output_tensor[:, :, [4089, 4091, 4093, 4094]] = 1j * input_tensor[:, :, 682].unsqueeze(-1)  # Make values imaginary
        return output_tensor
    def kern5ImagTransformer(self,input_tensor):
        #batch_size,300,682 k5s3p0
        N, embedding_dim, num_filters = self.batch_size, 300, 682

        # Step 1: Create an output tensor of zeros with shape (N, 300, 4096), as complex type
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex64)

        # Step 2: Assign imaginary values for outlier filter 0
        output_tensor[:, :, [1, 3, 5]] = 1j * input_tensor[:, :, 0].unsqueeze(-1)

        #[,1][,3][,5][6,][8,]
        #[,7][,9][,11][12,][14,]
        #[,13] [,15] [,17] [18,] [20,]
        #[,19][,21][,23][24,][26,]
        for i in range(1, 682):
            indices = [
                6 * (i - 1) + 1,
                6 * (i - 1) + 3,
                6 * (i - 1) + 5,
                6 * (i - 1) + 6,
                6 * (i - 1) + 8
            ]
            output_tensor[:, :, indices] = 1j * input_tensor[:, :, i].unsqueeze(-1)

        return output_tensor


class   cnnToLSTMCustomInterleaving(nn.Module):
    def __init__(self,vocab_size:   int , embedding_dim:    int , pretrained_vecs ,batch_size:    int , max_len:    int):
        super(cnnToLSTMCustomInterleaving,self).__init__()
        #top k2 k4
        #range(0,256,1)
        self.max_len = max_len
        self.embed  =   nn.Embedding(vocab_size, embedding_dim,dtype=torch.float16)
        print(self.embed.weight.data.size())
        self.embed.weight.data.copy_(pretrained_vecs)
        self.embed.weight.requires_grad = False
        self.batch_size = batch_size
        self.num_components = 512

        self.kern2s1 =   nn.Conv1d(in_channels=300,out_channels=300,kernel_size=2,stride=1,dtype=torch.float16) #255 to 2047
        self.kern4s2 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=4, stride=2,dtype=torch.float16)#127 to 1023
        #mid k3 k6
        self.kern3s3p1 =   nn.Conv1d(in_channels=300,out_channels=300,kernel_size=3,stride=3, padding=2,dtype=torch.float16)#(2045+2*2)/3+1=684 from 86
        self.kern6s3p1 =   nn.Conv1d(in_channels=300,out_channels=300,kernel_size=6,stride=3, padding=2,dtype=torch.float16)#(2042+2*2)/3+1 = 683 from 85
        #bottom k4
        self.kern5s3 =   nn.Conv1d(in_channels=300,out_channels=300,kernel_size=5,stride=3,padding=0,dtype=torch.float16)#(2043)/3+1=682  from 84

        self.LSTM_hidden    =   1024
        self.uppLSTM = nn.LSTM(600, 2048, batch_first=True, bidirectional=True,dtype=torch.float16)  #300*2 for interleaved
        self.midLSTM = nn.LSTM(600, 2048, batch_first=True, bidirectional=True,dtype=torch.float16)  #300*2 for interleaved

        self.lowLSTM = nn.LSTM(600, 2048, batch_first=True, bidirectional=True,dtype=torch.float16)  #no complex values
        self.weights    =   nn.Parameter(torch.tensor([0.25,0.25,0.25,0.25],dtype=torch.float16))

        self.fc1    =   nn.Linear(2048,16)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(32,2)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)

        # CNN Layers
        topk2 = self.kern2ImagTransformer(self.kern2s1(x))#torch.float16
        topk4 = self.kern4ImagTransformer(self.kern4s2(x))
        midk3 = self.kern3ImagTransformer(self.kern3s3p1(x))
        midk6 = self.kern6ImagTransformer(self.kern6s3p1(x))
        lowk5 = self.kern5ImagTransformer(self.kern5s3(x))

        def interleave_complex(complex_tensor):

            real_part = complex_tensor.real
            imag_part = complex_tensor.imag
            print(f"Real Part Shape: {real_part.shape}")
            batch_size, channels, seq_len = real_part.shape

            # Create interleaved tensor
            interleaved = torch.zeros(batch_size, channels * 2,seq_len,
                                      device=real_part.device, dtype=torch.float16)

            # Interleave real and imaginary parts
            interleaved[:, 0::2,:] = real_part
            interleaved[:, 1::2,:] = imag_part
            return interleaved

        # Process complex combinations with interleaving

        upper_combined = torch.tensor(topk2) + torch.tensor(topk4)
        upper_input = interleave_complex(upper_combined).transpose(1, 2)
        #16,600,1024
        mid_combined = torch.tensor(midk3) + torch.tensor(midk6)
        mid_input = interleave_complex(mid_combined).transpose(1, 2)
        #16,600,1024

        low_input = interleave_complex(torch.tensor(lowk5)).transpose(1, 2)
        # Process through LSTMs
        print(f"LSTM INPUT SHAPES: {upper_input.shape, mid_input.shape, low_input.shape}")
        #upp_out, _ = checkpoint.checkpoint(self._forward_lstm, self.uppLSTM, upper_input)
        #mid_out, _ = checkpoint.checkpoint(self._forward_lstm, self.midLSTM, mid_input)
        #low_out, _ = checkpoint.checkpoint(self._forward_lstm, self.lowLSTM, low_input)
        upp_out =   self.uppLSTM(upper_input)
        mid_out = self.midLSTM(mid_input)
        low_out = self.lowLSTM(low_input)

        def apply_pca(self, features,num_comp,chunk_size):
            mean = torch.mean(features, dim=0, keepdim=True)
            centered_data = features - mean
            N,D = centered_data.shape
            U_total,S_total,Vt_total = [],[],[]#torch.svd(centered_data)
            for start   in range(0,N,chunk_size):
                end =   min(start + chunk_size,N)
            cov_matrix = torch.matmul(centered_data.T, centered_data) / (features.shape[0] - 1)

            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

            sorted_indices = torch.argsort(eigenvalues, descending=True)
            top_eigenvectors = eigenvectors[:, sorted_indices[:self.num_components]]

            return torch.matmul(centered_data, top_eigenvectors)
        def apply_pla(features):
            features = features.mean(dim=1)

            chunk_size = 1024
            if features.shape[1] > chunk_size:
                cov_matrix = torch.zeros(features.shape[1], features.shape[1],
                                         device=features.device)
                for i in range(0, features.shape[1], chunk_size):
                    end = min(i + chunk_size, features.shape[1])
                    chunk = features[:, i:end]
                    cov_matrix[i:end] = chunk.T @ chunk
            else:
                cov_matrix = features.T @ features

            eigvals, eigvecs = torch.linalg.eigh(cov_matrix)

            # Sort and select top components
            sorted_indices = torch.argsort(eigvals, descending=True)
            top_k_eigvecs = eigvecs[:, sorted_indices[:self.num_components]]

            return features @ top_k_eigvecs
        # Apply PLA to LSTM outputs
        upp_features = apply_pla(upp_out)
        mid_features = apply_pla(mid_out)
        low_features = apply_pla(low_out)

        # Combine PLA-reduced features
        fused = upp_features + mid_features + low_features

        # Fully Connected Layers
        swisher = F.silu(self.fc1(fused))
        dropout = self.dropout(swisher)
        final_outputs = F.softmax(self.fc2(dropout), dim=1)
        return final_outputs

        # noinspection PyUnreachableCode

    def _forward_lstm(self,lstm,x):
        return  lstm(x)


    def kern2ImagTransformer(self,  input_tensor):
        #batch_size,300,2047   k2s1
        N, seq_len, num_filters = self.batch_size, 300, 2047
        ## Create index mapping for placement
        #indices = torch.arange(255).unsqueeze(0) * 2 + 1  # Calculate (2i+1)
        #indices = indices.repeat(N, seq_len, 1)  # Repeat for batch and sequence

        # Create the output tensor
        output_tensor = torch.zeros(N, seq_len, 4096, dtype=torch.float16)

        for i in range(num_filters):
            #[orig,overlap_back]
            # For each filter, map to positions 2i+1 and 2i+2
            output_tensor[:, :, 2*i+1] = input_tensor[:, :, i]
            output_tensor[:, :, 2*i+2] = input_tensor[:, :, i]


        odds    =   output_tensor[:,:,1::2]
        print(f"k2 Odds: {odds.shape}")
        return  odds
    def kern4ImagTransformer(self,input_tensor):
        #batch_size,300,1023  k4s2
        N, embedding_dim, num_filters = self.batch_size, 300, 1023  # Example sizes
        input_tensor=input_tensor.to(dtype=torch.complex32)
        output_tensor = torch.zeros(N, embedding_dim, 2048 * 2,dtype=torch.complex32)
        for i in range(num_filters):
            # Compute target indices for filter i
            indices = [4*i+1, 4*i+3, 4*i+4, 4*i+6]
            # Assign the input filter values as imaginary numbers to the output at the computed indices
            output_tensor[:, :, indices] = 1j * input_tensor[:, :, i].unsqueeze(-1)

        # split into odds
        odds    =   output_tensor[:,:,1::2]
        print(f"k4 Odds: {odds.shape}")
        return  odds
    def kern3ImagTransformer(self,input_tensor):
        #batch_size,300,684     k3s3p2
        N, embedding_dim, num_filters = self.batch_size, 300, 684  # Example sizes
        input_tensor=input_tensor.to(dtype=torch.complex32)
        output_tensor = torch.zeros(N, embedding_dim, 2048 * 2,dtype=torch.complex32)
        #[][][,1]
        #[,3][,5][,7]
        #[][][]
        #[][][]
        # clip off the first filter at index 0
        #output_tensor[:, :, [1]] = input_tensor[:, :, 0].unsqueeze(-1)
        for i in range(1, 683):
            indices = [6*i-3, 6*i-1, 6*i+1]#4089,4091,4093
            output_tensor[:, :, indices] = input_tensor[:, :, i].unsqueeze(-1)

        #cut off outlier filter at index 683
#        output_tensor[:, :, [4095]] = input_tensor[:, :, 683].unsqueeze(-1)
        odds    =   output_tensor[:,:,1::2]
        print(f"k3 Odds: {odds.shape}")
        return  odds
    def kern6ImagTransformer(self,input_tensor):

        # Original tensor of shape (N, 300, 683)
        #batch_size,300,683 k6s3p2
        N, embedding_dim, num_filters = self.batch_size, 300, 683  # Example sizes
        input_tensor=input_tensor.to(dtype=torch.complex32)
        output_tensor = torch.zeros(N, embedding_dim, 2048 * 2, dtype=torch.complex32)
        # [][][,1][2,][4,][6,]
        # [,3][,5][,7][8,][10,][12,]
        # [][][][][][]
        # [][][][][][]
        # Outlier filter 0
        output_tensor[:, :, [1, 2, 4, 6]] = 1j * input_tensor[:, :, 0].unsqueeze(-1)  # Make values imaginary
        # Regular filters 1 to 682
        for i in range(1, 682):#3,5,7, 8 , 10, 12
            indices = [6*i-3, 6*i-1, 6*i+1, 6*i+2, 6*i+4,6*i+6]
            output_tensor[:, :, indices] = 1j * input_tensor[:, :, i].unsqueeze(-1).repeat(1, 1, 6)  # Make values imaginary
#12:58.  1/23/25
        # Outlier filter _4083, _4085, _4087, 4088_, 4090_, 4092_
        # Outlier filter _4089, _4091, _4093, 4094_, __, __

        output_tensor[:, :, [4089, 4091, 4093, 4094]] = 1j * input_tensor[:, :, 682].unsqueeze(-1)  # Make values imaginary

        #split to odds
        odds    =   output_tensor[:,:,1::2]
        print(f"k6 Odds: {odds.shape}")
        return  odds
    def kern5ImagTransformer(self,input_tensor):
        #batch_size,300,682 k5s3p0
        N, embedding_dim, num_filters = self.batch_size, 300, 682

        # Step 1: Create an output tensor of zeros with shape (N, 300, 4096), as complex type
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex32)

        # Step 2: Assign imaginary values for outlier filter 0
        output_tensor[:, :, [1, 3, 5]] = 1j * input_tensor[:, :, 0].unsqueeze(-1)

        #[,1][,3][,5][6,][8,]
        #[,7][,9][,11][12,][14,]
        #[,13] [,15] [,17] [18,] [20,]
        #[,19][,21][,23][24,][26,]
        for i in range(1, 682):
            indices = [
                6 * (i - 1) + 1,
                6 * (i - 1) + 3,
                6 * (i - 1) + 5,
                6 * (i - 1) + 6,
                6 * (i - 1) + 8
            ]
            output_tensor[:, :, indices] = 1j * input_tensor[:, :, i].unsqueeze(-1)

#split to odds
        odds    =   output_tensor[:,:,1::2]
        print(f"k5 Odds: {odds.shape}")
        return  odds


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
                return torch.tensor(1, dtype=torch.float16)
            elif label == "neg":
                return torch.tensor(0, dtype=torch.float16)
            else:
                raise ValueError(f"Unexpected label: {label}")
        elif isinstance(label, int) or label.isdigit():
            return torch.tensor(float(int(label) - 1), dtype=torch.float16)
        else:
            raise ValueError(f"Unsupported label type: {label}")
    def pipeline_driver(raw_data_split):
        print(f"Max Len:     {max([len(text_pipeline(text)) for _, text in raw_data_split])}")
        #print(f"   {max([len(text_pipeline(text)) for _, text in raw_data_split])}")
        print(f"# of too big:{sum(len(text_pipeline(text)) > 2048 for _, text in raw_data_split)}")
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
max_len = 2048#realistitcally 2752
padding_type = 'post'
vocab_size = 130000#to be changed later
embedding_dim = 300
#max token


# hypers
batch_size = 16
epoch_count = 7
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
(train_dSet, val_dSet, test_dSet), (train_data, val_data, test_data) = process_dataset(combined_dataset)#take out the big ones




#filter out samples with text lengths greater than 2048 tokens
def filter_large_samples(dataset, tokenizer, max_tokens=2048):
    filtered_data = []
    for item in dataset:
        label,text = item  # Assuming each item is a (text, label) tuple
        tokenized_text = tokenizer(text)  # Tokenize the text
        #print(tokenized_text)
        if len(tokenized_text) <= max_tokens:
            filtered_data.append(item)
    return filtered_data

# Applying the filtering function to each dataset
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")  # Use a tokenizer that fits your dataset
train_dSet_filtered = filter_large_samples(train_dSet, tokenizer)
val_dSet_filtered = filter_large_samples(val_dSet, tokenizer)
test_dSet_filtered = filter_large_samples(test_dSet, tokenizer)

# Wrap the filtered datasets into PyTorch Dataset objects
train_dSet = redoneTupleDataset(train_dSet_filtered)
val_dSet = redoneTupleDataset(val_dSet_filtered)
test_dSet = redoneTupleDataset(test_dSet_filtered)


#filter out samples with text lengths greater than 2048 tokens
def filter_large_samples_regular(data, tokenizer, max_tokens=2048):

    return [(label, text) for label, text in data if len(text) <= max_tokens]


train_data_filtered = filter_large_samples_regular(train_data, tokenizer)
val_data_filtered = filter_large_samples_regular(val_data, tokenizer)
test_data_filtered = filter_large_samples_regular(test_data, tokenizer)

# Replace original data with filtered data
train_data = train_data_filtered
val_data = val_data_filtered
test_data = test_data_filtered








vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>", "<pad>"])
vocab_size = 130000#int(len(vocab.get_stoi()) // 2 + 1)
vocab.set_default_index(vocab["<unk>"])


glove_path = os.path.expanduser(
    "C:\\Users\\epw268\\Documents\\GitHub\\realtime-reddit-sentiments\\.vector_cache\\glove.6B.300d.txt")  # Adjust for your cache path
GloVe_itos = []


# stoi = {word: idx for idx, word in enumerate(GloVe_itos)}  # String-to-index mapping


# Simulate a vocabulary of size `vocab_size`
# Assuming the vocabulary is sorted by frequency (common practice in NLP tasks)
# "<unk>" and "<pad>" are added for unknown tokens and padding
print(dir(glove))
pad_idx = vocab["<pad>"]
# vocab_list = ["<pad>", "<unk>"] + list(stoi.keys())[:vocab_size - 2]

# Create vocab-to-index mapping
# word_to_index = {word: idx for idx, word in enumerate(vocab_list)}
# absurdly big auauauaua 10000000
pretrained_vectors = torch.zeros((vocab_size, embedding_dim))
# fix the vocab and use glove pretraining
print(f"Max idx: {max(vocab.get_stoi().values())}")
print(f"pretrained_vectors: {pretrained_vectors.shape}")
for word, idx in vocab.get_stoi().items():
    if word in glove.stoi:  # Check if word is in GloVe's vocabulary
        # pretrained_vectors[idx] = stoi[word]
        pretrained_vectors[idx] = torch.tensor(glove.stoi[word], dtype=torch.float16)
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
    max_length = 2048#4096
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


dLoad_train = DataLoader(train_dSet, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_batch,pin_memory=False)
dLoad_val = DataLoader(val_dSet, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_batch,pin_memory=False)
dLoad_test = DataLoader(test_dSet, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_batch,pin_memory=False)
# inst_test = IMDB(split="test")

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
#print(all_texts)
text_shapes =   [text.shape   for  text in all_texts]
#print(text_shapes)
#dim_problems

train_texts_tensor = torch.cat(all_texts, dim=0)
train_labels_tensor = torch.cat(all_labels, dim=0) - 1
print(f"Average Label Mag: {torch.mean(train_labels_tensor.float())}")

#print(str(type(dLoad_train)) + ".trainer    |.    " + str(dir(dLoad_train)))
#print(str(type(dLoad_test)) + ".    |.    " + str(dir(dLoad_test)))

"""def yield_token(data_iter):
    for _, text in data_iter:
        yield token_retriever(text)"""

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnnToLSTMCustomInterleaving(vocab_size, 300, pretrained_vectors,
                                    batch_size, int(2048)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with progress tracking
print(f"Starting training on {device}")
model.train()
for epoch in range(epoch_count):
    epoch_loss = 0
    progress_bar = tqdm(dLoad_train, desc=f'Epoch {epoch + 1}/{epoch_count}')

    for batch_idx, (labels, inputs) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}',
                                  'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'})

    print(f'Epoch {epoch + 1}/{epoch_count}, Average Loss: {epoch_loss / len(dLoad_train):.4f}')

# Validation with progress tracking
print("\nStarting validation...")
model.eval()
val_loss = 0
val_accuracy = 0
with torch.no_grad():
    for inputs, labels in tqdm(dLoad_val, desc='Validation'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_accuracy += ((outputs > 0.5) == labels).float().mean().item()

val_loss /= len(dLoad_val)
val_accuracy /= len(dLoad_val)
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Test evaluation with progress tracking
print("\nStarting test evaluation...")
test_accuracy = 0
with torch.no_grad():
    for inputs, labels in tqdm(dLoad_test, desc='Testing'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_accuracy += ((outputs > 0.5) == labels).float().mean().item()

test_accuracy /= len(dLoad_test)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), 'sentiment_model.pth')