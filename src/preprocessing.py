"""Text preprocessing utilities for the NovelForge EDA.

The goal of this module is to keep the notebook focused on analysis while the
reusable cleaning logic lives in plain Python code.
"""

from __future__ import annotations

import html
import ast
import math
import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MULTISPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*\b")
_LABEL_SPLIT_RE = re.compile(r"\s*(?:,|;|\||/)\s*")


def _is_missing_scalar(value: object) -> bool:
    """Return True for scalar missing values without triggering pandas typing noise."""
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _load_wordnet_lemmatizer():
    """Return an NLTK WordNet lemmatizer when available, otherwise None."""
    try:
        from nltk.stem import WordNetLemmatizer

        return WordNetLemmatizer()
    except Exception:
        return None


def _fallback_lemma(token: str) -> str:
    """Apply a conservative rule-based fallback when NLTK is unavailable."""
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


@dataclass
class TextPreprocessor:
    """Robust text cleaner for synopsis fields."""

    lowercase: bool = True
    remove_urls: bool = True
    lemmatize: bool = True

    def __post_init__(self) -> None:
        self._lemmatizer = _load_wordnet_lemmatizer() if self.lemmatize else None

    def clean_text(self, value: object) -> str:
        """Clean a single synopsis value and return a normalized string."""
        if _is_missing_scalar(value):
            return ""

        text = html.unescape(str(value))
        text = _HTML_TAG_RE.sub(" ", text)

        if self.remove_urls:
            text = _URL_RE.sub(" ", text)

        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = _MULTISPACE_RE.sub(" ", text).strip()

        if self.lowercase:
            text = text.lower()

        if self.lemmatize:
            text = self._lemmatize_text(text)

        return text

    def _lemmatize_text(self, text: str) -> str:
        """Lemmatize alphabetic tokens while preserving readable spacing."""
        tokens = _TOKEN_RE.findall(text)
        if not tokens:
            return ""

        if self._lemmatizer is not None:
            try:
                return " ".join(self._lemmatizer.lemmatize(token) for token in tokens)
            except LookupError:
                pass

        return " ".join(_fallback_lemma(token) for token in tokens)

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Clean a pandas Series of synopsis values."""
        return series.apply(self.clean_text)


def infer_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    """Infer a column name from common aliases.

    Raises a helpful ValueError when no candidate is found.
    """
    columns_list = list(columns)
    normalized = {column.lower().strip(): column for column in columns_list}

    for candidate in candidates:
        key = candidate.lower().strip()
        if key in normalized:
            return normalized[key]

    raise ValueError(
        "No expected column found. Available columns: "
        + ", ".join(columns_list)
        + ". Expected one of: "
        + ", ".join(candidates)
    )


def drop_columns_if_present(
    df: pd.DataFrame,
    columns_to_drop: Iterable[str],
) -> pd.DataFrame:
    """Drop optional columns without failing when one is absent."""
    existing_columns = [column for column in columns_to_drop if column in df.columns]
    return df.drop(columns=existing_columns).copy()


def parse_multilabel_cell(value: object) -> list[str]:
    """Normalize a multilabel cell into a list of clean labels."""
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        if _is_missing_scalar(value):
            return []

        text = str(value).strip()
        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
            raw_items = parsed if isinstance(parsed, (list, tuple, set)) else _LABEL_SPLIT_RE.split(text)
        except (ValueError, SyntaxError):
            raw_items = _LABEL_SPLIT_RE.split(text)

    return [str(item).strip() for item in raw_items if str(item).strip()]


def filter_labels(
    labels: Iterable[str],
    allowed_labels: Iterable[str],
) -> list[str]:
    """Keep labels that belong to an allowed taxonomy, preserving canonical names."""
    canonical_by_key = {label.lower().strip(): label for label in allowed_labels}
    filtered: list[str] = []

    for label in labels:
        key = str(label).lower().strip()
        if key in canonical_by_key and canonical_by_key[key] not in filtered:
            filtered.append(canonical_by_key[key])

    return filtered


def add_filtered_label_column(
    df: pd.DataFrame,
    label_column: str,
    output_column: str,
    allowed_labels: Iterable[str],
) -> pd.DataFrame:
    """Parse and filter a raw multilabel column into a genre taxonomy column."""
    result = df.copy()
    result[output_column] = result[label_column].apply(
        lambda value: filter_labels(parse_multilabel_cell(value), allowed_labels)
    )
    return result


def add_text_features(
    df: pd.DataFrame,
    text_column: str,
    clean_column: str = "synopsis_clean",
) -> pd.DataFrame:
    """Add word and character length features for descriptive statistics."""
    result = df.copy()
    result["word_count"] = result[clean_column].str.split().str.len().fillna(0).astype(int)
    result["char_count"] = result[clean_column].str.len().fillna(0).astype(int)
    result["raw_char_count"] = result[text_column].fillna("").astype(str).str.len()
    return result


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str,
    clean_column: str = "synopsis_clean",
    preprocessor: TextPreprocessor | None = None,
) -> pd.DataFrame:
    """Clean the synopsis column and add standard text features."""
    cleaner = preprocessor or TextPreprocessor()
    result = df.copy()
    result[clean_column] = cleaner.clean_series(result[text_column])
    return add_text_features(result, text_column=text_column, clean_column=clean_column)


def detect_synopsis_anomalies(
    df: pd.DataFrame,
    clean_column: str = "synopsis_clean",
    min_words: int = 5,
) -> pd.DataFrame:
    """Return rows with empty or suspiciously short cleaned synopsis values."""
    word_count = df[clean_column].str.split().str.len().fillna(0).astype(int)
    mask = df[clean_column].fillna("").str.strip().eq("") | word_count.lt(min_words)

    anomalies = df.loc[mask].copy()
    anomalies["anomaly_reason"] = "empty_or_too_short_synopsis"
    anomalies["word_count"] = word_count.loc[mask]
    return anomalies


def remove_synopsis_anomalies(
    df: pd.DataFrame,
    clean_column: str = "synopsis_clean",
    min_words: int = 5,
) -> pd.DataFrame:
    """Remove empty or too-short synopsis rows."""
    word_count = df[clean_column].str.split().str.len().fillna(0).astype(int)
    mask = df[clean_column].fillna("").str.strip().ne("") & word_count.ge(min_words)
    return df.loc[mask].copy()
