import torch
from torch.utils.data import Dataset, ConcatDataset
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
    # Calculate dataset sizes
    total_size = len(combined_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    # Split the dataset
    train_dSet, val_dSet, test_dSet = torch.utils.data.random_split(
        combined_dataset,
        [train_size, val_size, test_size]
    )

    return (train_dSet, val_dSet, test_dSet)


def main():
    # Load IMDB dataset from Hugging Face
    imdb_dataset = load_dataset('stanfordnlp/imdb')

    # Prepare tokenizer (you can choose a different one if needed)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Wrap datasets
    train_wrapped = RedoneTupleDataset(imdb_dataset['train'])
    test_wrapped = RedoneTupleDataset(imdb_dataset['test'])

    # Combine datasets
    combined_dataset = ConcatDataset([train_wrapped, test_wrapped])

    # Embedding dimension (adjust as needed)
    embedding_dim = 300

    # Load GloVe embeddings
    glove = GloVe(name='6B', dim=embedding_dim)

    # Process dataset
    train_dSet, val_dSet, test_dSet = process_dataset(combined_dataset)

    # Optional: Print dataset sizes
    print(f"Train dataset size: {len(train_dSet)}")
    print(f"Validation dataset size: {len(val_dSet)}")
    print(f"Test dataset size: {len(test_dSet)}")

    return train_dSet, val_dSet, test_dSet, glove


if __name__ == '__main__':
    train_data, val_data, test_data, glove_embeddings = main()