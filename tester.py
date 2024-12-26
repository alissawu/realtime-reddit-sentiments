import torch
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
    return torch.tensor(vocab(tokenizer(text)), dtype=torch.int64)

# Helper function to process labels to tensor
def label_pipeline(label):
    return torch.tensor(int(label) - 1, dtype=torch.int64)  # Convert to 0 and 1 for binary classification

# Load train and test data with transformations
def process_dataset(split):
    raw_iter = IMDB(root=".data", split=split)
    data = [
        (label_pipeline(label), text_pipeline(text))
        for label, text in raw_iter
    ]
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

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_batch)

# Example: Iterate through the DataLoader
for labels, texts, lengths in train_loader:
    print("Labels:", labels)
    print("Texts:", texts)
    print("Lengths:", lengths)
    break
