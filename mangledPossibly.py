import torch
import os
from huggingface_hub import get_token
from torch.utils.data import Dataset, ConcatDataset
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from datasets import load_dataset
from transformers import AutoTokenizer

class RedoneTupleDataset(Dataset):
    def __init__(self, original_dataset):
        self.data = []
        for item in original_dataset:
            self.data.append((item['text'], item['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def process_dataset(combined_dataset, val_split=0.2, test_split=0.1):
    total_size = len(combined_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    train_dSet, val_dSet, test_dSet = torch.utils.data.random_split(
        combined_dataset,
        [train_size, val_size, test_size]
    )

    return (train_dSet, val_dSet, test_dSet)


def main():
    imdb_dataset = load_dataset('stanfordnlp/imdb')

    tokenizer   =   get_tokenizer('basic_english')

    train_wrapped = RedoneTupleDataset(imdb_dataset['train'])
    test_wrapped = RedoneTupleDataset(imdb_dataset['test'])
    combined_dataset = ConcatDataset([train_wrapped, test_wrapped])
    # params
    max_len = 2048  # realistitcally 2752
    padding_type = 'post'
    vocab_size = 130000  # to be changed later
    embedding_dim = 300
    # max token

    # hypers
    batch_size = 16
    epoch_count = 7
    learning_rate = 0.004
    min_lr = 0.0005
    #GloVe embeddings
    glove = GloVe(name='6B', dim=embedding_dim)
    glove_path = os.path.expanduser(
        "C:\\Users\\epw268\\Documents\\GitHub\\realtime-reddit-sentiments\\.vector_cache\\glove.6B.300d.txt")  # Adjust for your cache path
    GloVe_itos = []
    #simulate a vocab of size vocab_size
    #assume vocab is sorted by frequency
    # '<unk>' and '<pad>'
    print(dir(glove))

    vocab   =   
    pad_idx = vocab['<pad']


    train_dSet, val_dSet, test_dSet = process_dataset(combined_dataset)

    print(f"Train dataset size: {len(train_dSet)}")
    print(f"Validation dataset size: {len(val_dSet)}")
    print(f"Test dataset size: {len(test_dSet)}")

    return train_dSet, val_dSet, test_dSet, glove


if __name__ == '__main__':
    train_data, val_data, test_data, glove_embeddings = main()