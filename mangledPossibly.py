import torch
import os
from huggingface_hub import get_token
from torch.utils.data import Dataset, ConcatDataset
from torchtext.data.utils import get_tokenizer  # Updated import
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


class TextPipeline:
    def __init__(self, glove_embeddings):
        self.vocab = {word: idx for idx, word in enumerate(glove_embeddings.itos)}
        self.unk_token_idx = self.vocab.get('<unk>', 0)
        self.pad_token_idx = self.vocab.get('<pad>', 1)
        self.max_length = 2048

    def __call__(self, text):
        tokens = [self.vocab.get(word.lower(), self.unk_token_idx)
                  for word in text.split()][:self.max_length]
        return torch.tensor(tokens, dtype=torch.long)

    def pad_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        if len(tokens) < self.max_length:
            padding = torch.full((self.max_length - len(tokens),),
                                 self.pad_token_idx,
                                 dtype=torch.long)
            return torch.cat([tokens, padding])
        return tokens[:self.max_length]


class LabelPipeline:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def __call__(self, label: int):
        return torch.tensor(label, dtype=torch.long)

    def one_hot(self, label):
        return torch.nn.functional.one_hot(
            self.__call__(label),
            self.num_classes
        )


def process_dataset(combined_dataset, val_split=0.2, test_split=0.1):
    # Set generator for reproducibility
    generator = torch.Generator().manual_seed(42)

    total_size = len(combined_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    train_dSet, val_dSet, test_dSet = torch.utils.data.random_split(
        combined_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_dSet, val_dSet, test_dSet


def yield_tokens(data_iter):
    tokenizer = get_tokenizer("basic_english")
    for text, _ in data_iter:
        if isinstance(text, str):
            yield tokenizer(text)
        elif isinstance(text, list):
            yield text
        else:
            raise ValueError("Unexpected text format. Expected string or list of tokens.")


from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, ConcatDataset


def main():
    # Load IMDB dataset using torchtext
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')

    # Convert iterators to lists for easier handling
    train_data = list(train_iter)
    test_data = list(test_iter)

    # Create wrapped datasets
    train_wrapped = RedoneTupleDataset(train_data)
    test_wrapped = RedoneTupleDataset(test_data)
    combined_dataset = ConcatDataset([train_wrapped, test_wrapped])

    # Rest of your code remains the same
    max_len = 2048
    padding_type = 'post'
    vocab_size = 130000
    embedding_dim = 300

    batch_size = 16
    epoch_count = 7
    learning_rate = 0.004
    min_lr = 0.0005

    glove = GloVe(name='6B', dim=embedding_dim)

    # Create vocabulary mapping
    vocab = {word: idx for idx, word in enumerate(glove.itos)}

    if '<pad>' not in vocab:
        vocab['<pad>'] = len(vocab)
    if '<unk>' not in vocab:
        vocab['<unk>'] = len(vocab)

    pad_idx = vocab['<pad>']

    train_dSet, val_dSet, test_dSet = process_dataset(combined_dataset)

    print(f"Train dataset size: {len(train_dSet)}")
    print(f"Validation dataset size: {len(val_dSet)}")
    print(f"Test dataset size: {len(test_dSet)}")

    return train_dSet, val_dSet, test_dSet, glove


# You might need to modify your RedoneTupleDataset class slightly:
class RedoneTupleDataset(Dataset):
    def __init__(self, original_dataset):
        self.data = []
        for label, text in original_dataset:  # torchtext IMDB returns (label, text) tuples
            self.data.append((text, label))  # Swap order to match your expected format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    train_data, val_data, test_data, glove_embeddings = main()