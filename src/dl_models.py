"""Deep learning utilities for NovelForge multilabel genre prediction."""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, hamming_loss, jaccard_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*\b")


def tokenize_text(text: object) -> list[str]:
    """Tokenize a cleaned synopsis into lowercase word tokens."""
    return TOKEN_RE.findall(str(text).lower())


@dataclass
class TextVocabulary:
    """Simple word-level vocabulary for LSTM experiments."""

    max_vocab_size: int = 30_000
    min_freq: int = 2
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    def __post_init__(self) -> None:
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
        self.id_to_token = [self.pad_token, self.unk_token]

    def fit(self, texts: Iterable[str]) -> "TextVocabulary":
        """Build a vocabulary from training texts only."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize_text(text))

        max_words = max(0, self.max_vocab_size - len(self.id_to_token))
        for token, freq in counter.most_common():
            if freq < self.min_freq:
                continue
            if len(self.id_to_token) >= max_words + 2:
                break
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)

        return self

    def transform_one(self, text: str, max_length: int) -> list[int]:
        """Convert one text into a padded/truncated sequence of token IDs."""
        unk_id = self.token_to_id[self.unk_token]
        ids = [self.token_to_id.get(token, unk_id) for token in tokenize_text(text)]
        ids = ids[:max_length]
        if len(ids) < max_length:
            ids.extend([0] * (max_length - len(ids)))
        return ids

    def transform(self, texts: Iterable[str], max_length: int) -> np.ndarray:
        """Convert texts into a padded integer matrix."""
        return np.asarray([self.transform_one(text, max_length) for text in texts], dtype=np.int64)

    @property
    def size(self) -> int:
        """Return vocabulary size."""
        return len(self.id_to_token)


class TextMultilabelDataset(Dataset):
    """PyTorch Dataset for padded text sequences and multilabel targets."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


class LSTMGenreClassifier(nn.Module):
    """Embedding + LSTM classifier with sigmoid-ready multilabel logits."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * direction_factor, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return raw logits for BCEWithLogitsLoss."""
        embedded = self.embedding(input_ids)
        _, (hidden_state, _) = self.lstm(embedded)

        if self.lstm.bidirectional:
            sequence_repr = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
        else:
            sequence_repr = hidden_state[-1]

        return self.classifier(self.dropout(sequence_repr))


@dataclass
class LSTMTrainingConfig:
    """Hyperparameters for the LSTM baseline."""

    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    bidirectional: bool = True
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 8
    patience: int = 2
    threshold: float = 0.5
    max_length: int = 160
    max_vocab_size: int = 30_000
    min_freq: int = 2


def get_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloader(
    sequences: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for text multilabel data."""
    dataset = TextMultilabelDataset(sequences, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_lstm_model(
    model: LSTMGenreClassifier,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: LSTMTrainingConfig,
    device: torch.device | None = None,
    pos_weight: torch.Tensor | None = None,
) -> tuple[LSTMGenreClassifier, list[dict[str, float]]]:
    """Train an LSTM with Adam and EarlyStopping on validation F1 micro."""
    device = device or get_device()
    model = model.to(device)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_state = None
    best_valid_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        model.train()
        train_loss = 0.0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)

        train_loss /= len(train_loader.dataset)
        valid_metrics = evaluate_lstm_model(model, valid_loader, config.threshold, device=device)
        epoch_seconds = time.perf_counter() - start_time

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "valid_f1_micro": valid_metrics["f1_micro"],
            "valid_f1_macro": valid_metrics["f1_macro"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        if valid_metrics["f1_micro"] > best_valid_f1:
            best_valid_f1 = valid_metrics["f1_micro"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict_proba_lstm(
    model: LSTMGenreClassifier,
    data_loader: DataLoader,
    device: torch.device | None = None,
) -> np.ndarray:
    """Predict multilabel probabilities."""
    device = device or get_device()
    model = model.to(device)
    model.eval()
    probabilities = []

    with torch.no_grad():
        for input_ids, _ in data_loader:
            input_ids = input_ids.to(device)
            probs = torch.sigmoid(model(input_ids))
            probabilities.append(probs.cpu().numpy())

    return np.vstack(probabilities)


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Compute BCE positive class weights for imbalanced multilabel targets."""
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    weights = negatives / np.maximum(positives, 1)
    return torch.as_tensor(weights, dtype=torch.float32)


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Iterable[float] | None = None,
    average: str = "micro",
) -> tuple[float, float]:
    """Find the threshold that maximizes F1 on validation probabilities."""
    thresholds = thresholds or np.arange(0.10, 0.55, 0.05)
    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, y_pred, average=average, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def evaluate_lstm_model(
    model: LSTMGenreClassifier,
    data_loader: DataLoader,
    threshold: float = 0.5,
    target_names: list[str] | None = None,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Evaluate an LSTM with the same multilabel metrics as the ML baseline."""
    y_true = data_loader.dataset.labels.numpy().astype(int)
    y_proba = predict_proba_lstm(model, data_loader, device=device)
    y_pred = (y_proba >= threshold).astype(int)

    metrics: dict[str, object] = {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "classification_report_text": classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
        ),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics
