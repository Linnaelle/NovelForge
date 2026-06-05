"""Baseline multilabel model for NovelForge.

This module keeps the classical ML baseline outside the notebook so the
experiment remains reproducible and easy to reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


@dataclass
class BaselineModel:
    """TF-IDF + regularized Logistic Regression multilabel baseline."""

    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    penalty: Literal["l1", "l2"] = "l2"
    C: float = 1.0
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "lbfgs"
    max_iter: int = 1_000
    random_state: int = 42

    def __post_init__(self) -> None:
        penalty = self.penalty.lower()
        if penalty not in {"l1", "l2"}:
            raise ValueError("penalty must be either 'l1' or 'l2'.")

        l1_ratio = 0.0 if penalty == "l2" else 1.0

        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=self.ngram_range,
                        min_df=self.min_df,
                        max_df=self.max_df,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=self.C,
                            l1_ratio=l1_ratio,
                            solver=self.solver,
                            max_iter=self.max_iter,
                            class_weight="balanced",
                            random_state=self.random_state,
                        )
                    ),
                ),
            ]
        )

    def train(self, X_train: pd.Series | list[str], y_train: Any) -> "BaselineModel":
        """Fit the TF-IDF vectorizer and multilabel classifier."""
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: pd.Series | list[str]):
        """Predict binary multilabel outputs."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.Series | list[str]):
        """Predict multilabel probabilities when the classifier supports it."""
        return self.pipeline.predict_proba(X)

    def evaluate(
        self,
        X_test: pd.Series | list[str],
        y_test: Any,
        target_names: list[str] | None = None,
        X_train: pd.Series | list[str] | None = None,
        y_train: Any | None = None,
    ) -> dict[str, Any]:
        """Evaluate multilabel performance and optional bias/variance signals."""
        y_pred = self.predict(X_test)

        metrics: dict[str, Any] = {
            "test_f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
            "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "test_f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "test_jaccard_samples": jaccard_score(
                y_test,
                y_pred,
                average="samples",
                zero_division=0,
            ),
            "test_hamming_loss": hamming_loss(y_test, y_pred),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
            ),
            "classification_report_text": classification_report(
                y_test,
                y_pred,
                target_names=target_names,
                zero_division=0,
            ),
        }

        if X_train is not None and y_train is not None:
            y_train_pred = self.predict(X_train)
            train_f1_micro = f1_score(y_train, y_train_pred, average="micro", zero_division=0)
            train_f1_macro = f1_score(y_train, y_train_pred, average="macro", zero_division=0)

            metrics.update(
                {
                    "train_f1_micro": train_f1_micro,
                    "train_f1_macro": train_f1_macro,
                    "generalization_gap_micro": train_f1_micro - metrics["test_f1_micro"],
                    "generalization_gap_macro": train_f1_macro - metrics["test_f1_macro"],
                }
            )

        return metrics


def apply_thresholds(probabilities: Any, thresholds: float | list[float]):
    """Convert multilabel probabilities to binary predictions."""
    return (probabilities >= thresholds).astype(int)


def find_best_global_threshold(
    y_true: Any,
    probabilities: Any,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, float]:
    """Find the global decision threshold that maximizes micro F1."""
    candidate_thresholds = thresholds or [value / 100 for value in range(10, 71, 5)]
    best_threshold = 0.5
    best_score = -1.0

    for threshold in candidate_thresholds:
        y_pred = apply_thresholds(probabilities, threshold)
        score = f1_score(y_true, y_pred, average="micro", zero_division=0)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)

    return best_threshold, best_score


def find_best_label_thresholds(
    y_true: Any,
    probabilities: Any,
    thresholds: Iterable[float] | None = None,
) -> list[float]:
    """Find one F1-optimized threshold per label."""
    candidate_thresholds = list(thresholds or [value / 100 for value in range(10, 71, 5)])
    label_thresholds: list[float] = []

    for label_index in range(probabilities.shape[1]):
        best_threshold = 0.5
        best_score = -1.0

        for threshold in candidate_thresholds:
            y_pred = (probabilities[:, label_index] >= threshold).astype(int)
            score = f1_score(y_true[:, label_index], y_pred, zero_division=0)
            if score > best_score:
                best_threshold = float(threshold)
                best_score = float(score)

        label_thresholds.append(best_threshold)

    return label_thresholds
