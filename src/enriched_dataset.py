"""Dataset assembly helpers for the enriched NovelForge experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.preprocessing import (
    TextPreprocessor,
    add_text_features,
    clean_dataframe,
    drop_columns_if_present,
    parse_multilabel_cell,
    remove_synopsis_anomalies,
)
from src.project_config import ENRICHED_GENRE_VOCABULARY, ENRICHED_LABEL_MAPPING


def normalize_labels(labels: Iterable[str]) -> list[str]:
    """Map raw platform labels to the enriched taxonomy."""
    normalized: list[str] = []

    for label in labels:
        key = str(label).lower().strip().replace("_", " ")
        canonical = ENRICHED_LABEL_MAPPING.get(key)
        if canonical and canonical not in normalized:
            normalized.append(canonical)

    return normalized


def split_pipe_labels(value: object) -> list[str]:
    """Parse MyAnimeList pipe-separated label columns."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    return [item.strip() for item in str(value).split("|") if item.strip()]


def _join_non_empty(values: Iterable[object]) -> list[str]:
    labels: list[str] = []
    for value in values:
        labels.extend(split_pipe_labels(value))
    return labels


def load_original_dataset(data_path: Path) -> pd.DataFrame:
    """Load the current project dataset and map its tags to the enriched taxonomy."""
    raw = pd.read_csv(data_path)
    raw = drop_columns_if_present(raw, ["cover"])
    raw["genre_labels"] = raw["tags"].apply(lambda value: normalize_labels(parse_multilabel_cell(value)))
    raw["source_dataset"] = "novelforge_current"
    raw["media_type"] = "manga_lightnovel"

    cleaner = TextPreprocessor(lowercase=True, remove_urls=True, lemmatize=True)
    df = clean_dataframe(raw, text_column="description", clean_column="synopsis_clean", preprocessor=cleaner)
    df = remove_synopsis_anomalies(df, clean_column="synopsis_clean", min_words=5)
    df = df[df["genre_labels"].str.len().gt(0)].copy()
    return df[["title", "synopsis_clean", "genre_labels", "source_dataset", "media_type"]]


def load_mal_dataset(csv_path: Path, media_type: str) -> pd.DataFrame:
    """Load an anime or manga MAL dataset and map genres/themes/demographics."""
    raw = pd.read_csv(csv_path)
    raw["genre_labels"] = raw.apply(
        lambda row: normalize_labels(
            _join_non_empty([row.get("genres", ""), row.get("themes", ""), row.get("demographics", "")])
        ),
        axis=1,
    )
    raw["source_dataset"] = csv_path.stem
    raw["media_type"] = media_type

    cleaner = TextPreprocessor(lowercase=True, remove_urls=True, lemmatize=True)
    df = clean_dataframe(raw, text_column="synopsis", clean_column="synopsis_clean", preprocessor=cleaner)
    df = remove_synopsis_anomalies(df, clean_column="synopsis_clean", min_words=5)
    df = df[df["genre_labels"].str.len().gt(0)].copy()
    return df[["title", "synopsis_clean", "genre_labels", "source_dataset", "media_type"]]


def build_enriched_dataset(
    project_dir: Path,
    include_original: bool = True,
    include_manga: bool = True,
    include_anime: bool = True,
) -> pd.DataFrame:
    """Build the combined dataset used by the before/after experiment."""
    frames: list[pd.DataFrame] = []

    if include_original:
        frames.append(load_original_dataset(project_dir / "data" / "data.csv"))

    archive_dir = project_dir / "data" / "archive (1)"
    if include_manga:
        frames.append(load_mal_dataset(archive_dir / "manga_dataset.csv", media_type="manga"))
    if include_anime:
        frames.append(load_mal_dataset(archive_dir / "anime_dataset.csv", media_type="anime"))

    if not frames:
        raise ValueError("At least one dataset source must be enabled.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["synopsis_clean"]).copy()
    combined["label_count"] = combined["genre_labels"].str.len()
    return combined


def summarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return label counts for a dataframe with a genre_labels column."""
    counts = {label: 0 for label in ENRICHED_GENRE_VOCABULARY}
    for labels in df["genre_labels"]:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    return (
        pd.DataFrame({"label": list(counts.keys()), "count": list(counts.values())})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
