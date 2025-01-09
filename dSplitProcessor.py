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

def process_dataset():
    # Load the IMDB dataset
    train_iter = IMDB(root=".data", split='train')  # Load the 'train' split

    # Convert to list for random splitting
    data = [(label, text) for label, text in train_iter]

    # Compute sizes for train and initial test splits
    total_size = len(data)
    train_size = total_size * 0.7
    val_size    = total_size * 0.2
    test_size = total_size * 0.1
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for label, text in data_iter:
            yield tokenizer(text)

    # Load train split to build vocabulary
    train_iter = IMDB(root=".data", split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # Helper function to process text to tensor
    def text_pipeline(text):
        return torch.tensor(vocab(tokenizer(text)), dtype=torch.int64)

    # Helper function to process labels to tensor
    def label_pipeline(label):
        if isinstance(label, str):
            if label == "pos":
                return torch.tensor(1, dtype=torch.float)
            elif label == "neg":
                return torch.tensor(0, dtype=torch.float)
            else:
                raise ValueError(f"Unexpected label: {label}")
        elif isinstance(label, int) or label.isdigit():
            return torch.tensor(int(label) - 1, dtype=torch.float)
        else:
            raise ValueError(f"Unsupported label type: {label}")
        # Load train and test data with transformations
#FROM HERE STARTTT
    def pipeline_driver(raw_data_split):
        return  [(label_pipeline(label),text_pipeline(text))
                 for label, text in raw_data_split
        ]

    # Create train and test datasets with random split
    train_data, test_data,val_data = random_split(data, [train_size,val_size, test_size])
    #train_size =   25000+25000//split_1 or 50000*0.7
    #test_size   =   25000(1-1//split_1) or 50000*0.1
    #val_size   =   25000(8/10-2/5) or 50000*0.2
    # Split the test_data further into train_data, val_data, and test

    # Preprocess all datasets using label and text pipelines
    train_data  = pipeline_driver(train_data)
    val_data    = pipeline_driver(val_data)
    test_data   = pipeline_driver(test_data)
    return train_data, val_data, test_data
    # Print sizes of each split

    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Testing data size: {len(test_data)}")