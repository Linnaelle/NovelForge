"""Visualization helpers for the NovelForge EDA notebook."""

from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


GENRE_SPLIT_RE = re.compile(r"\s*(?:,|;|\||/)\s*")


def set_plot_style() -> None:
    """Apply a clean, consistent visual style for notebook charts."""
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="viridis",
        rc={
            "figure.figsize": (10, 6),
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def _parse_genres(value: object) -> list[str]:
    """Normalize a genre cell into a list of genre labels."""
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        if value is None or pd.isna(value):
            return []

        text = str(value).strip()
        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                raw_items = parsed
            else:
                raw_items = GENRE_SPLIT_RE.split(text)
        except (ValueError, SyntaxError):
            raw_items = GENRE_SPLIT_RE.split(text)

    return [str(item).strip() for item in raw_items if str(item).strip()]


def count_genres(series: pd.Series) -> pd.Series:
    """Count genres in a multi-label genre column."""
    counter: Counter[str] = Counter()
    for value in series:
        counter.update(_parse_genres(value))

    return pd.Series(counter).sort_values(ascending=False)


def plot_top_genres(
    df: pd.DataFrame,
    genre_column: str,
    top_n: int = 15,
    title: str = "Top 15 genres",
):
    """Plot a polished horizontal bar chart for the most frequent genres."""
    set_plot_style()
    genre_counts = count_genres(df[genre_column]).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax, orient="h")

    ax.set_title(title, weight="bold", pad=14)
    ax.set_xlabel("Number of titles")
    ax.set_ylabel("")

    max_value = genre_counts.max() if not genre_counts.empty else 0
    for index, value in enumerate(genre_counts.values):
        ax.text(value + max_value * 0.01, index, f"{int(value)}", va="center", fontsize=10)

    fig.tight_layout()
    return fig, ax


def plot_synopsis_length_distribution(
    df: pd.DataFrame,
    length_column: str = "word_count",
    bins: int = 40,
    title: str = "Distribution of synopsis length",
):
    """Plot a histogram of synopsis lengths with a mean reference line."""
    set_plot_style()
    lengths = df[length_column].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(lengths, bins=bins, kde=True, ax=ax, color="#2a9d8f")

    mean_length = lengths.mean()
    median_length = lengths.median()
    ax.axvline(mean_length, color="#e76f51", linestyle="--", linewidth=2, label=f"Mean: {mean_length:.1f}")
    ax.axvline(median_length, color="#264653", linestyle=":", linewidth=2, label=f"Median: {median_length:.1f}")

    ax.set_title(title, weight="bold", pad=14)
    ax.set_xlabel("Words per synopsis")
    ax.set_ylabel("Number of titles")
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig, ax


def plot_missing_values(
    df: pd.DataFrame,
    title: str = "Missing values by column",
):
    """Plot missing value percentages for columns with at least one missing value."""
    set_plot_style()
    missing_rate = df.isna().mean().mul(100).sort_values()
    missing_rate = missing_rate[missing_rate.gt(0)]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(missing_rate))))
    if missing_rate.empty:
        ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        sns.barplot(x=missing_rate.values, y=missing_rate.index, ax=ax, orient="h", color="#457b9d")
        ax.set_xlabel("Missing values (%)")
        ax.set_ylabel("")
        ax.set_xlim(0, min(100, max(5, missing_rate.max() * 1.15)))

    ax.set_title(title, weight="bold", pad=14)
    fig.tight_layout()
    return fig, ax
