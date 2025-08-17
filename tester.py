#pytorch 2.6
#python 3.11
import re
import gensim.downloader as api
print("Downloading 300D GloVe vectors...")
#glove_vectors = api.load("glove-wiki-gigaword-100")


def basic_tokenizer(text: str) -> list[str]:
    # Lowercase the text
    text = text.lower()
    # Add spaces around punctuation we want to isolate
    for punct in ["'", ".", ",", "(", ")", "!", "?", ";", ":"]:
        text = text.replace(punct, f" {punct} ")
    text = text.replace('"', "")           # remove double quotes
    text = text.replace("<br />", " ")     # replace HTML line breaks with space
    # Split on whitespace
    tokens = text.split()
    return tokens

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torch.amp import autocast, GradScaler
from contextlib import  nullcontext
# Updated imports for newer versions
#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import #build_vocab_from_iterator, GloVe


torch.backends.cudnn.benchmark = True


from typing import Callable, Sequence, Tuple
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class Vocab:
    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {idx: tok for tok, idx in stoi.items()}
        self.default_index = stoi["<unk>"] if "<unk>" in stoi else None
    def __len__(self):
        return len(self.stoi)
    def __getitem__(self, token: str):
        # Return index if in vocab, else the default <unk>
        if token in self.stoi:
            return self.stoi[token]
        elif self.default_index is not None:
            return self.default_index
        else:
            raise KeyError(f"Token '{token}' not in vocab")
    def get_stoi(self):
        return self.stoi
    def set_default_index(self, idx: int):
        self.default_index = idx
    def __call__(self, tokens: list[str]) -> list[int]:
        # Convert list of tokens to list of indices
        return [self.__getitem__(tok) for tok in tokens]


class BatchCollator:
    """
    Picklable collate function for DataLoader workers
    tokenizes text
    - Maps tokens to vocab indices (LongTensor for nn.Embedding)
    - Truncates/pads to exactly max_len
    - Returns (input_ids [B, max_len] long, labels [B] float)
    """
    def __init__(
        self,
        vocab: Vocab,
        max_len: int,
        tokenizer: Callable[[str], list[str]] = basic_tokenizer,
        pad_token: str = "<pad>",
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pad_idx = vocab.stoi.get(pad_token, 0)

    def __call__(self, batch: Sequence[Tuple[str, int]]):
        texts, labels = zip(*batch)

        seq_tensors = []
        for text in texts:
            # tokenize -> ids
            ids = [self.vocab[tok] for tok in self.tokenizer(text)]
            # truncate early to cap memory/compute
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            seq_tensors.append(torch.tensor(ids, dtype=torch.long))

        # pad to batch max first
        padded = pad_sequence(seq_tensors, batch_first=True, padding_value=self.pad_idx)

        # then force to exactly max_len (right-pad or clip)
        if padded.size(1) < self.max_len:
            pad_len = self.max_len - padded.size(1)
            right_pad = torch.full((padded.size(0), pad_len), self.pad_idx, dtype=torch.long)
            padded = torch.cat([padded, right_pad], dim=1)
        elif padded.size(1) > self.max_len:
            padded = padded[:, :self.max_len]

        labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
        return padded, labels_tensor


class CNNToLSTMCustomInterleaving(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, pretrained_vecs, batch_size: int, max_len: int,
                 device=None):
        super().__init__()
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_components = embedding_dim
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
        self.weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32, device=self.device))

        # PCA components buffer
        D = embedding_dim * 2
        self.register_buffer('pca_mean', torch.zeros(D, dtype=torch.float32, device=self.device))
        self.register_buffer('pca_components', torch.zeros(D, self.num_components, dtype=torch.float32, device=self.device))
        self.pca_fitted = False

        # Final layers
        self.fc1 = nn.Linear(embedding_dim, 256).to(self.device)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 16).to(self.device)
        self.fc3 = nn.Linear(16, 1).to(self.device)

    def _no_autocast_ctx(self):
        try:
            return torch.cuda.amp.autocast(enabled=False) if self.device.type == "cuda" else nullcontext()
        except Exception:
            return nullcontext()
    def fit_pca(self, features):
        """Fit PCA on batch of features and writes to registered buffers
        W/O creating graph history"""

        with torch.no_grad(), self._no_autocast_ctx():

            batch, seq_len, D = features.shape
            flattened = features.reshape(-1, D).to(self.device, dtype=torch.float32)

            # Compute mean
            #self.pca_mean = flattened.mean(dim=0)
            mu = flattened.mean(dim=0)
            #self.pca_mean.copy_(mu)#keep registered buffer f32

            centered = flattened - mu
            denom = max(centered.shape[0] - 1, 1) #numerically safe denom

            # Compute covariance
            cov_matrix =  (torch.matmul(centered.T , centered) / denom).to(self.device, dtype=torch.float32)#torch.matmul(centered.T, centered) / (flattened.shape[0] - 1)

            # Eigendecomposition
            e_vals, e_vecs = torch.linalg.eigh(cov_matrix)
            sorted_indices = torch.argsort(e_vals, descending=True)
            comps = e_vecs[:, sorted_indices[:self.num_components]]

            #keeps buffers registered THEN copies into them
            self.pca_mean.copy_(mu)
            self.pca_components.copy_(comps)
            self.pca_fitted = True


    def apply_pca(self, features):
        """Apply PCA transformation"""
        with autocast(device_type=self.device.type, enabled=False):
            batch, seq_len, D = features.shape
            flattened = features.reshape(-1, D).to(self.device, dtype=torch.float32)

            if not self.pca_fitted:
                self.fit_pca(features)

            centered = flattened - self.pca_mean
            projected = torch.matmul(centered ,self.pca_components)
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
            output_tensor[:, :, 2 * i + 1:2 * i + 3] = (input_tensor[:, :, i:i + 1].to(torch.complex64))

        return output_tensor

    def kern4ImagTransformer(self, input_tensor):
        N, embedding_dim, num_filters = input_tensor.shape
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex64, device=self.device)

        for i in range(min(num_filters, 1023)):
            indices = [4 * i + 1, 4 * i + 3, 4 * i + 4, 4 * i + 6]
            valid_indices = [idx for idx in indices if idx < 4096]
            if valid_indices:
                output_tensor[:, :, valid_indices] = (1j * input_tensor[:, :, i].unsqueeze(-1)).to(output_tensor.dtype)

        return output_tensor

    def kern3ImagTransformer(self, input_tensor):
        N, embedding_dim, num_filters = input_tensor.shape
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex64, device=self.device)

        for i in range(1, min(num_filters - 1, 683)):
            indices = [6 * i - 3, 6 * i - 1, 6 * i + 1]
            valid_indices = [idx for idx in indices if 0 <= idx < 4096]
            if valid_indices:
                #output_tensor[:, :, valid_indices] = input_tensor[:, :, i].unsqueeze(-1)
                output_tensor[:, :, valid_indices] = (input_tensor[:, :, i].unsqueeze(-1).to(torch.complex64))
        return output_tensor

    def kern6ImagTransformer(self, input_tensor):
        N, embedding_dim, num_filters = input_tensor.shape
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex64, device=self.device)

        # Handle first filter
        first_indices = [1, 2, 4, 6]
        output_tensor[:, :, first_indices] = (1j * input_tensor[:, :, 0].unsqueeze(-1)).to(output_tensor.dtype)

        # Regular filters
        for i in range(1, min(num_filters - 1, 682)):
            indices = [6 * i - 3, 6 * i - 1, 6 * i + 1, 6 * i + 2, 6 * i + 4, 6 * i + 6]
            valid_indices = [idx for idx in indices if 0 <= idx < 4096]
            if valid_indices:
                output_tensor[:, :, valid_indices] = (1j * input_tensor[:, :, i].unsqueeze(-1)).to(output_tensor.dtype)

        return output_tensor

    def kern5ImagTransformer(self, input_tensor):
        N, embedding_dim, num_filters = input_tensor.shape
        output_tensor = torch.zeros(N, embedding_dim, 4096, dtype=torch.complex64, device=self.device)

        # First filter
        output_tensor[:, :, [1, 3, 5]] = (1j * input_tensor[:, :, 0].unsqueeze(-1)).to(output_tensor.dtype)

        # Regular filters
        for i in range(1, min(num_filters, 682)):
            indices = [6 * (i - 1) + 1, 6 * (i - 1) + 3, 6 * (i - 1) + 5, 6 * (i - 1) + 6, 6 * (i - 1) + 8]
            valid_indices = [idx for idx in indices if 0 <= idx < 4096]
            if valid_indices:
                output_tensor[:, :, valid_indices] = (1j * input_tensor[:, :, i].unsqueeze(-1)).to(output_tensor.dtype)

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

"""
def load_glove_embeddings(vocab, embedding_dim=100, cache_dir=None):
    if cache_dir is None:
        cache_dir = Path.home() / '.vector_cache'

    # Try to load GloVe
    try:
        glove = GloVe(name='6B', dim=embedding_dim, cache=str(cache_dir))
    except:
        print("Downloading GloVe embeddings...")
        glove = GloVe(name='6B', dim=embedding_dim)

    # Create embedding matrix
    embedding_dim = 100  # using 100d GloVe
    vocab_size = len(vocab)
    pretrained_embeddings = torch.zeros((vocab_size, embedding_dim), dtype=torch.float32)
    for word, idx in vocab.get_stoi().items():
        if word in glove_vectors:
            pretrained_embeddings[idx] = torch.tensor(glove_vectors[word], dtype=torch.float32)
        elif word == "<pad>":
            pretrained_embeddings[idx] = torch.zeros(embedding_dim)  # pad -> zero vector
        else:
            # OOV token: random vector
            pretrained_embeddings[idx] = torch.randn(embedding_dim) * 0.1

    return pretrained_embeddings
"""


def load_glove_embeddings(vocab: Vocab,
                          embedding_dim: int = 300,
                          cache_dir: str | Path | None = None,
                          lower_fallback: bool = True,
                          oov_std: float = 0.1) -> torch.Tensor:
    """
    Returns [len(vocab), embedding_dim] Float32 tensor init from gensim GloVe.
    Unknowns are ~N(0, oov_std^2); <pad> is all zeros

    vocab: Vocab instance (must expose .stoi and get_stoi())
    embedding_dim: {50, 100, 200, 300} for wiki-gigaword GloVe
    cache_dir: optional dir for gensim downloader cache
    lower_fallback: if token not found, try token.lower()
    oov_std: normal std for random init of OOV rows
    """
    #give gensim loc to cache @ BEFORE LOADING
    if cache_dir is not None:
        os.environ.setdefault("GENSIM_DATA_DIR", str(cache_dir))

    #map dim -> gensim model name
    name_map = {
        50: "glove-wiki-gigaword-50",
        100: "glove-wiki-gigaword-100",
        200: "glove-wiki-gigaword-200",
        300: "glove-wiki-gigaword-300",
    }
    if embedding_dim not in name_map:
        raise ValueError(f"embedding_dim must be one of {sorted(name_map)}")

    #downloads then uses local cache
    kv = api.load(name_map[embedding_dim])  #gensim KeyedVectors

    vocab_size = len(vocab)
    emb = torch.empty(vocab_size, embedding_dim, dtype=torch.float32)
    torch.nn.init.normal_(emb, mean=0.0, std=oov_std)  #OOV default

    #make <pad> zero vector if present
    pad_idx = vocab.stoi.get("<pad>")
    if pad_idx is not None:
        emb[pad_idx].zero_()

    #fast membership via key_to_index; pull vectors with get_vector()
    key_to_index = kv.key_to_index
    for token, idx in vocab.get_stoi().items():
        if token == "<pad>":
            continue
        if token in key_to_index:
            emb[idx] = torch.from_numpy(kv.get_vector(token))
        elif lower_fallback and token.lower() in key_to_index:
            emb[idx] = torch.from_numpy(kv.get_vector(token.lower()))
        # else: keep rand init
    return emb

def create_data_loaders(batch_size=16, max_len=2048, vocab_size=130000):
    """Create data loaders with proper preprocessing"""
    # Load datasets
    print("Loading IMDB dataset...")
    train_data = load_dataset('stanfordnlp/imdb', split='train')
    test_data = load_dataset('stanfordnlp/imdb', split='test')

    # Combine and prepare data
    all_data = []
    for item in train_data:
        all_data.append((item['text'], item['label']))
    for item in test_data:
        all_data.append((item['text'], item['label']))

    # Initialize tokenizer
    tokenizer = basic_tokenizer

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

    from collections import Counter
    counter = Counter()
    for text, _ in train_data:  # iterate over training split
        tokens = basic_tokenizer(text)
        counter.update(tokens)

    # Build vocabulary of most common tokens
    special_tokens = ["<unk>", "<pad>"]
    vocab_size = 130000
    most_common = counter.most_common(vocab_size - len(special_tokens))
    # Start vocabulary with special tokens
    stoi = {tok: idx for idx, tok in enumerate(special_tokens)}
    for token, _ in most_common:
        if token not in stoi:  # avoid any overlap with special tokens
            stoi[token] = len(stoi)

    vocab = Vocab(stoi)
    vocab.set_default_index(vocab.stoi["<unk>"])

    # Collate function
    def collate_batch(batch):
        texts, labels = zip(*batch)
        # Tokenize each text and convert to indices
        tokenized = [basic_tokenizer(text) for text in texts]
        numericalized = [torch.tensor(vocab(tokens), dtype=torch.float32) for tokens in tokenized]
        # Pad sequences to max_len
        padded = pad_sequence(numericalized, batch_first=True, padding_value=vocab["<pad>"])
        if padded.size(1) > max_len:
            padded = padded[:, :max_len]
        elif padded.size(1) < max_len:
            # pad to the right if shorter than max_len
            pad_length = max_len - padded.size(1)
            padding_tensor = torch.full((padded.size(0), pad_length), vocab["<pad>"], dtype=padded.dtype)
            padded = torch.cat([padded, padding_tensor], dim=1)
        # Convert labels to float tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return padded, labels_tensor
    # Create data loaders




    collate = BatchCollator(vocab=vocab, max_len=max_len, tokenizer=basic_tokenizer)
    NUM_W = _suggest_num_workers()

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=NUM_W,
        pin_memory=True,  #enables async H2D w/ non_blocking
        persistent_workers=True,  #keep workers alive across epochs
        prefetch_factor=4  #each worker preloads 4 batches worth
    )

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=max(2, NUM_W // 2), pin_memory=True, persistent_workers=True, prefetch_factor=2)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=max(2, NUM_W // 2), pin_memory=True, persistent_workers=True, prefetch_factor=2)

    return train_loader, val_loader, test_loader, vocab, tokenizer

def _suggest_num_workers() -> int:
    import os
    #leave 1-2 CPUs for main process; cap at 8 to avoid oversubscription
    return max(2, min((os.cpu_count() or 4) - 2, 8))


class CUDAPrefetcher:
    """
    Overlaps H2D copies with compute via dedicated CUDA stream
    Yields (inputs, labels) already on device with non_blocking copies.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.next_batch = None

    def __iter__(self):
        it = iter(self.loader)
        self._prefetch(it)
        while self.next_batch is not None:
            if self.stream is not None:
                torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.next_batch
            self._prefetch(it)
            yield batch

    def _prefetch(self, it):
        try:
            data, target = next(it)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:  #cpu‑only path
            self.next_batch = (data, target)
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = (
                data.to(self.device, non_blocking=True),
                target.to(self.device, non_blocking=True),
            )


def get_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """Choose most efficient opt avail"""
    try:
        from apex.optimizers import FusedAdam
        print("Using NVIDIA Apex FusedAdam optimizer.")
        return FusedAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    except ImportError:
        try:
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
            print("Apex not found; using PyTorch AdamW with fused=True.")
            return opt
        except TypeError:
            # Fused not supported in this build, use standard AdamW
            print("WARNING: AdamW(fused=True) not supported, falling back to standard AdamW.")
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def train_model(model, train_loader, val_loader, epochs=7, lr=0.004, device="cuda"):
    """Training function with mixed precision support"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model, lr=lr, weight_decay=0.01)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    warmup_steps = max(1, warmup_steps)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps]
                                               )
    #step scheduler each batch
    # Mixed precision training
    scaler = GradScaler()

    best_val_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        amp_enabled = (device.type == "cuda")

        prefetch = CUDAPrefetcher(train_loader, device)
        progress_bar = tqdm(prefetch, desc=f"Epoch {epoch + 1}/{epochs}")

        #progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for inputs, labels in progress_bar:
            if device.type != "cuda":  # CPU fall-back
                inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            #mixed precision forward pass
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            #backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            #statistics
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
            for inputs, labels in tqdm(CUDAPrefetcher(val_loader, device), desc="Validation"):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                with autocast(device_type=device.type, enabled=amp_enabled):
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

        print(f"\nEpoch {epoch + 1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, "best_model.pth")


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    current_file_path = Path(__file__)
    cache_dir = (Path(__file__).parent / "glove_vecs").as_posix()
    pretrained_vectors = load_glove_embeddings(vocab, embedding_dim, cache_dir, True, 0.25)

    """
    def load_glove_embeddings(vocab: Vocab,
                          embedding_dim: int = 100,
                          cache_dir: str | Path | None = None,
                          lower_fallback: bool = True,
                          oov_std: float = 0.1) -> torch.Tensor:
    """
    # Create model
    model = CNNToLSTMCustomInterleaving(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        pretrained_vecs=pretrained_vectors,
        batch_size=batch_size,
        max_len=max_len,
        device=device
    )

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
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
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
