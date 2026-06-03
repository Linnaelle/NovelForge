"""Lightweight HuggingFace Transformer utilities for NovelForge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, hamming_loss, jaccard_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class TransformerTextDataset(Dataset):
    """Dataset wrapping HuggingFace tokenized inputs and multilabel targets."""

    def __init__(self, encodings: dict[str, Any], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = labels.astype("float32")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


@dataclass
class TransformerConfig:
    """CPU-friendly defaults for a small multilabel Transformer experiment."""

    model_name: str = "distilbert-base-uncased"
    max_length: int = 160
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    epochs: int = 1
    weight_decay: float = 0.01
    threshold: float = 0.5
    output_dir: str = "models/transformer_novelforge"
    logging_steps: int = 25


class NovelForgeTransformer:
    """Small HuggingFace wrapper for multilabel genre classification."""

    def __init__(
        self,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        config: TransformerConfig | None = None,
    ) -> None:
        self.config = config or TransformerConfig()
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            problem_type="multi_label_classification",
        )

    def tokenize(self, texts: list[str] | np.ndarray) -> dict[str, list[list[int]]]:
        """Tokenize and truncate raw texts."""
        return self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.config.max_length,
        )

    def make_dataset(self, texts: list[str] | np.ndarray, labels: np.ndarray) -> TransformerTextDataset:
        """Create a Transformer dataset."""
        return TransformerTextDataset(self.tokenize(texts), labels)

    def fine_tune(
        self,
        train_texts: list[str] | np.ndarray,
        train_labels: np.ndarray,
        valid_texts: list[str] | np.ndarray,
        valid_labels: np.ndarray,
    ) -> Trainer:
        """Fine-tune the model lightly with HuggingFace Trainer."""
        train_dataset = self.make_dataset(train_texts, train_labels)
        valid_dataset = self.make_dataset(valid_texts, valid_labels)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
            greater_is_better=True,
            logging_steps=self.config.logging_steps,
            report_to=[],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self._compute_metrics,
        )
        trainer.train()
        self.model = trainer.model
        return trainer

    def predict_proba(self, texts: list[str] | np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """Predict multilabel probabilities."""
        dummy_labels = np.zeros((len(texts), self.num_labels), dtype="float32")
        dataset = self.make_dataset(texts, dummy_labels)
        args = TrainingArguments(
            output_dir=str(Path(self.config.output_dir) / "predict_tmp"),
            per_device_eval_batch_size=batch_size or self.config.eval_batch_size,
            report_to=[],
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )
        logits = trainer.predict(dataset).predictions
        return torch.sigmoid(torch.tensor(logits)).numpy()

    def evaluate(
        self,
        texts: list[str] | np.ndarray,
        labels: np.ndarray,
        target_names: list[str],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate with the same multilabel metrics used by the baseline."""
        threshold = threshold if threshold is not None else self.config.threshold
        probabilities = self.predict_proba(texts)
        predictions = (probabilities >= threshold).astype(int)

        return {
            "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
            "jaccard_samples": jaccard_score(labels, predictions, average="samples", zero_division=0),
            "hamming_loss": hamming_loss(labels, predictions),
            "classification_report_text": classification_report(
                labels,
                predictions,
                target_names=target_names,
                zero_division=0,
            ),
            "probabilities": probabilities,
            "predictions": predictions,
        }

    def save(self, output_dir: str | Path | None = None) -> None:
        """Persist model and tokenizer."""
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        id2label: dict[int, str],
        label2id: dict[str, int],
        config: TransformerConfig | None = None,
    ) -> "NovelForgeTransformer":
        """Load a local fine-tuned Transformer."""
        config = config or TransformerConfig()
        config.model_name = str(model_dir)
        return cls(num_labels=len(id2label), id2label=id2label, label2id=label2id, config=config)

    def _compute_metrics(self, prediction) -> dict[str, float]:
        labels = prediction.label_ids
        probabilities = torch.sigmoid(torch.tensor(prediction.predictions)).numpy()
        predictions = (probabilities >= self.config.threshold).astype(int)
        return {
            "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        }
