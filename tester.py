#pytorch 2.6
#python 3.11


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torch.amp import autocast, GradScaler

# Updated imports for newer versions
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class CNNToLSTMCustomInterleaving(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, pretrained_vecs, batch_size: int, max_len: int,
                 device=None):
        super().__init__()
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_components = 300
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize embedding layer with float32
        self.embed = nn.Embedding(vocab_size, embedding_dim, dtype=torch.float32).to(self.device)
        if pretrained_vecs is not None:
            pretrained_vecs = pretrained_vecs.to(self.device, dtype=torch.float32)
            self.embed.weight.data.copy_(pretrained_vecs)
        self.embed.weight.requires_grad = False

        # CNN layers
        self.kern2s1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=2, stride=1).to(self.device)
        self.kern4s2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=4, stride=2).to(self.device)
        self.kern3s3p1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=3, padding=2).to(self.device)
        self.kern6s3p1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=6, stride=3, padding=2).to(self.device)
        self.kern5s3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, stride=3, padding=0).to(self.device)

        # LSTM layers
        lstm_hidden = embedding_dim
        lstm_input_size = self.num_components
        self.uppLSTM = nn.LSTM(lstm_input_size, lstm_hidden, batch_first=True,
                               bidirectional=False, dropout=0.2).to(self.device)
        self.midLSTM = nn.LSTM(lstm_input_size, lstm_hidden, batch_first=True,
                               bidirectional=False, dropout=0.2).to(self.device)
        self.lowLSTM = nn.LSTM(lstm_input_size, lstm_hidden, batch_first=True,
                               bidirectional=False, dropout=0.25).to(self.device)

        # Learnable weights for fusion
        self.weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32))

        # PCA components buffer
        D = embedding_dim * 2
        self.register_buffer('pca_mean', torch.zeros(D, dtype=torch.float32))
        self.register_buffer('pca_components', torch.zeros(D, self.num_components, dtype=torch.float32))
        self.pca_fitted = False

        # Final layers
        self.fc1 = nn.Linear(embedding_dim, 256).to(self.device)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 16).to(self.device)
        self.fc3 = nn.Linear(16, 1).to(self.device)

    def fit_pca(self, features):
        """Fit PCA on a batch of features"""
        batch, seq_len, D = features.shape
        flattened = features.reshape(-1, D)

        # Compute mean
        self.pca_mean = flattened.mean(dim=0)
        centered = flattened - self.pca_mean

        # Compute covariance
        cov_matrix = torch.matmul(centered.T, centered) / (flattened.shape[0] - 1)

        # Eigendecomposition
        e_vals, e_vecs = torch.linalg.eigh(cov_matrix)
        sorted_indices = torch.argsort(e_vals, descending=True)
        self.pca_components = e_vecs[:, sorted_indices[:self.num_components]]
        self.pca_fitted = True

    def apply_pca(self, features):
        """Apply PCA transformation"""
        batch, seq_len, D = features.shape
        flattened = features.reshape(-1, D)

        if not self.pca_fitted:
            self.fit_pca(features)

        centered = flattened - self.pca_mean
        projected = torch.matmul(centered, self.pca_components)
        return projected.reshape(batch, seq_len, self.num_components)

    def interleave_complex(self, complex_tensor):
        """Interleave real and imaginary parts"""
        real_part = complex_tensor.real
        imag_part = complex_tensor.imag
        batch_size, channels, seq_len = real_part.shape

        # Create interleaved tensor
        interleaved = torch.zeros(batch_size, channels * 2, seq_len,
                                  device=self.device, dtype=torch.float32)
        interleaved[:, 0::2, :] = real_part
        interleaved[:, 1::2, :] = imag_part
        return interleaved

    def kern2ImagTransformer(self, input_tensor):
        N, embedding_dim, num_filters = input_tensor.shape
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex64, device=self.device)

        # Vectorized assignment
        for i in range(min(num_filters, 2047)):
            output_tensor[:, :, 2 * i + 1:2 * i + 3] = input_tensor[:, :, i:i + 1]

        return output_tensor

    def kern4ImagTransformer(self, input_tensor):
        N, embedding_dim, num_filters = input_tensor.shape
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex64, device=self.device)

        for i in range(min(num_filters, 1023)):
            indices = [4 * i + 1, 4 * i + 3, 4 * i + 4, 4 * i + 6]
            valid_indices = [idx for idx in indices if idx < 4096]
            if valid_indices:
                output_tensor[:, :, valid_indices] = 1j * input_tensor[:, :, i].unsqueeze(-1)

        return output_tensor

    def kern3ImagTransformer(self, input_tensor):
        N, embedding_dim, num_filters = input_tensor.shape
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex64, device=self.device)

        for i in range(1, min(num_filters - 1, 683)):
            indices = [6 * i - 3, 6 * i - 1, 6 * i + 1]
            valid_indices = [idx for idx in indices if 0 <= idx < 4096]
            if valid_indices:
                output_tensor[:, :, valid_indices] = input_tensor[:, :, i].unsqueeze(-1)

        return output_tensor

    def kern6ImagTransformer(self, input_tensor):
        N, embedding_dim, num_filters = input_tensor.shape
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex64, device=self.device)

        # Handle first filter
        first_indices = [1, 2, 4, 6]
        output_tensor[:, :, first_indices] = 1j * input_tensor[:, :, 0].unsqueeze(-1)

        # Regular filters
        for i in range(1, min(num_filters - 1, 682)):
            indices = [6 * i - 3, 6 * i - 1, 6 * i + 1, 6 * i + 2, 6 * i + 4, 6 * i + 6]
            valid_indices = [idx for idx in indices if 0 <= idx < 4096]
            if valid_indices:
                output_tensor[:, :, valid_indices] = 1j * input_tensor[:, :, i].unsqueeze(-1)

        return output_tensor

    def kern5ImagTransformer(self, input_tensor):
        N, embedding_dim, num_filters = input_tensor.shape
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex64, device=self.device)

        # First filter
        output_tensor[:, :, [1, 3, 5]] = 1j * input_tensor[:, :, 0].unsqueeze(-1)

        # Regular filters
        for i in range(1, min(num_filters, 682)):
            indices = [6 * (i - 1) + 1, 6 * (i - 1) + 3, 6 * (i - 1) + 5, 6 * (i - 1) + 6, 6 * (i - 1) + 8]
            valid_indices = [idx for idx in indices if 0 <= idx < 4096]
            if valid_indices:
                output_tensor[:, :, valid_indices] = 1j * input_tensor[:, :, i].unsqueeze(-1)

        return output_tensor

    def forward(self, x):
        # Ensure correct batch size
        current_batch_size = x.size(0)

        # Embedding
        x = self.embed(x)  # [batch, seq_len, embedding_dim]
        x = x.permute(0, 2, 1)  # [batch, embedding_dim, seq_len]
        embedding_mat = x

        # CNN transformations
        topk2 = self.kern2ImagTransformer(self.kern2s1(x))
        topk4 = self.kern4ImagTransformer(self.kern4s2(x))
        midk3 = self.kern3ImagTransformer(self.kern3s3p1(x))
        midk6 = self.kern6ImagTransformer(self.kern6s3p1(x))
        lowk5 = self.kern5ImagTransformer(self.kern5s3(x))

        # Combine and process
        upper_combined = topk2 + topk4
        upper_interleaved = self.interleave_complex(upper_combined).transpose(1, 2)
        upper_input = self.apply_pca(upper_interleaved)

        mid_combined = midk3 + midk6
        mid_interleaved = self.interleave_complex(mid_combined).transpose(1, 2)
        mid_input = self.apply_pca(mid_interleaved)

        low_interleaved = self.interleave_complex(lowk5).transpose(1, 2)
        low_input = self.apply_pca(low_interleaved)

        # LSTM processing
        upp_out, _ = self.uppLSTM(upper_input)
        mid_out, _ = self.midLSTM(mid_input)
        low_out, _ = self.lowLSTM(low_input)

        # Mean pooling over sequence dimension
        mean_lstm1 = upp_out.mean(dim=1)
        mean_lstm2 = mid_out.mean(dim=1)
        mean_lstm3 = low_out.mean(dim=1)
        mean_embed = embedding_mat.mean(dim=2)

        # Weighted fusion
        fused = (self.weights[0] * mean_lstm1 +
                 self.weights[1] * mean_lstm2 +
                 self.weights[2] * mean_lstm3 +
                 self.weights[3] * mean_embed)

        # Final layers
        x = F.silu(self.fc1(fused))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)

        return x


class IMDBDataset(Dataset):
    """Custom dataset wrapper for IMDB data"""

    def __init__(self, data):
        self.data = [(text, label) for text, label in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_glove_embeddings(vocab, embedding_dim=300, cache_dir=None):
    """Load GloVe embeddings with proper caching"""
    if cache_dir is None:
        cache_dir = Path.home() / '.vector_cache'

    # Try to load GloVe
    try:
        glove = GloVe(name='6B', dim=embedding_dim, cache=str(cache_dir))
    except:
        print("Downloading GloVe embeddings...")
        glove = GloVe(name='6B', dim=embedding_dim)

    # Create embedding matrix
    vocab_size = len(vocab)
    pretrained_vectors = torch.zeros((vocab_size, embedding_dim))

    for word, idx in vocab.get_stoi().items():
        if word in glove.stoi:
            pretrained_vectors[idx] = glove[word]
        elif word == "<pad>":
            pretrained_vectors[idx] = torch.zeros(embedding_dim)
        else:  # OOV words including <unk>
            pretrained_vectors[idx] = torch.randn(embedding_dim) * 0.1

    return pretrained_vectors


def create_data_loaders(batch_size=16, max_len=2048, vocab_size=130000):
    """Create data loaders with proper preprocessing"""
    # Load datasets
    print("Loading IMDB dataset...")
    train_data = load_dataset('imdb', split='train')
    test_data = load_dataset('imdb', split='test')

    # Combine and prepare data
    all_data = []
    for item in train_data:
        all_data.append((item['text'], item['label']))
    for item in test_data:
        all_data.append((item['text'], item['label']))

    # Initialize tokenizer
    tokenizer = get_tokenizer("basic_english")

    # Filter long sequences
    filtered_data = []
    for text, label in tqdm(all_data, desc="Filtering long sequences"):
        tokens = tokenizer(text)
        if len(tokens) <= max_len:
            filtered_data.append((text, label))

    print(f"Filtered {len(all_data) - len(filtered_data)} samples exceeding {max_len} tokens")

    # Split data
    total_size = len(filtered_data)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size

    # Create random splits
    train_data, val_data, test_data = random_split(
        filtered_data, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Build vocabulary
    print("Building vocabulary...")

    def yield_tokens(data_iter):
        for text, _ in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(train_data),
        specials=["<unk>", "<pad>"],
        special_first=True,
        max_tokens=vocab_size
    )
    vocab.set_default_index(vocab["<unk>"])

    # Collate function
    def collate_batch(batch):
        texts, labels = zip(*batch)

        # Tokenize and numericalize
        tokenized = [tokenizer(text) for text in texts]
        numericalized = [torch.tensor(vocab(tokens)) for tokens in tokenized]

        # Pad sequences
        padded = pad_sequence(numericalized, batch_first=True,
                              padding_value=vocab["<pad>"])

        # Truncate if necessary
        if padded.size(1) > max_len:
            padded = padded[:, :max_len]
        elif padded.size(1) < max_len:
            padding = torch.full((padded.size(0), max_len - padded.size(1)),
                                 vocab["<pad>"], dtype=padded.dtype)
            padded = torch.cat([padded, padding], dim=1)

        labels = torch.tensor(labels, dtype=torch.float32)
        return padded, labels

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_batch, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_batch, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_batch, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, vocab, tokenizer


def train_model(model, train_loader, val_loader, epochs=7, lr=0.004, device='cuda'):
    """Training function with mixed precision support"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision training
    scaler = GradScaler()

    best_val_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Statistics
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct / train_total:.4f}'
            })

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'\nEpoch {epoch + 1}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')

        scheduler.step()

    return model


def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    batch_size = 16
    max_len = 2048
    vocab_size = 130000
    embedding_dim = 300
    epochs = 7
    learning_rate = 0.004

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader, vocab, tokenizer = create_data_loaders(
        batch_size=batch_size, max_len=max_len, vocab_size=vocab_size
    )

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Load pretrained embeddings
    print("Loading pretrained embeddings...")
    pretrained_vectors = load_glove_embeddings(vocab, embedding_dim)

    # Create model
    model = CNNToLSTMCustomInterleaving(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        pretrained_vecs=pretrained_vectors,
        batch_size=batch_size,
        max_len=max_len,
        device=device
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    model = train_model(model, train_loader, val_loader, epochs=epochs,
                        lr=learning_rate, device=device)

    # Test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type='cuda'):
                outputs = model(inputs)

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f'Test Accuracy: {test_acc:.4f}')

    # Save final model
    torch.save(model.state_dict(), 'sentiment_model_final.pth')
    print("Model saved to sentiment_model_final.pth")


if __name__ == '__main__':
    main()