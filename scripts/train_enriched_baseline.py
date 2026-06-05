"""Train the enriched before/after TF-IDF baseline.

The evaluation keeps the current NovelForge dataset as the test domain:
- before: train only on the current dataset train split;
- after: train on the same current train split plus the enriched manga MAL rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score, hamming_loss, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.baseline_ml import (
    BaselineModel,
    apply_thresholds,
    find_best_global_threshold,
    find_best_label_thresholds,
)
from src.enriched_dataset import build_enriched_dataset, load_original_dataset, summarize_labels
from src.project_config import ENRICHED_GENRE_VOCABULARY


def evaluate_from_probabilities(y_true, probabilities, thresholds, labels: list[str]) -> dict:
    """Evaluate probabilities after applying fixed multilabel thresholds."""
    y_pred = apply_thresholds(probabilities, thresholds)
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
        "classification_report_text": classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0,
        ),
    }


def train_model(x_train: pd.Series, y_train) -> BaselineModel:
    """Train the shared optimized TF-IDF model."""
    model = BaselineModel(
        max_features=60_000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.92,
        C=1.5,
        solver="lbfgs",
        max_iter=1_000,
    )
    return model.train(x_train, y_train)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--include-anime", action="store_true")
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    models_dir = project_dir / "models"
    reports_dir = project_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    current = load_original_dataset(project_dir / "data" / "data.csv")
    enriched = build_enriched_dataset(project_dir, include_original=False, include_manga=True, include_anime=args.include_anime)

    current_train, current_temp = train_test_split(current, test_size=0.30, random_state=42, shuffle=True)
    current_valid, current_test = train_test_split(current_temp, test_size=0.50, random_state=42, shuffle=True)

    after_train = pd.concat([current_train, enriched], ignore_index=True)

    mlb = MultiLabelBinarizer(classes=ENRICHED_GENRE_VOCABULARY)
    mlb.fit([ENRICHED_GENRE_VOCABULARY])

    y_current_train = mlb.transform(current_train["genre_labels"])
    y_after_train = mlb.transform(after_train["genre_labels"])
    y_valid = mlb.transform(current_valid["genre_labels"])
    y_test = mlb.transform(current_test["genre_labels"])
    labels = list(mlb.classes_)

    print("Training before model on current dataset only...")
    before_model = train_model(current_train["synopsis_clean"], y_current_train)
    before_valid_proba = before_model.predict_proba(current_valid["synopsis_clean"])
    before_test_proba = before_model.predict_proba(current_test["synopsis_clean"])
    before_threshold, before_valid_f1 = find_best_global_threshold(y_valid, before_valid_proba)
    before_metrics = evaluate_from_probabilities(y_test, before_test_proba, before_threshold, labels)

    print("Training after model on current + MAL manga dataset...")
    after_model = train_model(after_train["synopsis_clean"], y_after_train)
    after_valid_proba = after_model.predict_proba(current_valid["synopsis_clean"])
    after_test_proba = after_model.predict_proba(current_test["synopsis_clean"])
    after_global_threshold, after_valid_f1 = find_best_global_threshold(y_valid, after_valid_proba)
    after_label_thresholds = find_best_label_thresholds(y_valid, after_valid_proba)

    after_global_metrics = evaluate_from_probabilities(y_test, after_test_proba, after_global_threshold, labels)
    after_label_metrics = evaluate_from_probabilities(y_test, after_test_proba, after_label_thresholds, labels)

    metrics_summary = pd.DataFrame(
        [
            {
                "model": "before_current_only",
                "train_rows": len(current_train),
                "external_rows": 0,
                "threshold_strategy": "global",
                "threshold": before_threshold,
                "valid_f1_micro": before_valid_f1,
                **{key: before_metrics[key] for key in ["f1_micro", "f1_macro", "f1_weighted", "jaccard_samples", "hamming_loss"]},
            },
            {
                "model": "after_current_plus_manga",
                "train_rows": len(after_train),
                "external_rows": len(enriched),
                "threshold_strategy": "global",
                "threshold": after_global_threshold,
                "valid_f1_micro": after_valid_f1,
                **{key: after_global_metrics[key] for key in ["f1_micro", "f1_macro", "f1_weighted", "jaccard_samples", "hamming_loss"]},
            },
            {
                "model": "after_current_plus_manga",
                "train_rows": len(after_train),
                "external_rows": len(enriched),
                "threshold_strategy": "per_label",
                "threshold": None,
                "valid_f1_micro": None,
                **{key: after_label_metrics[key] for key in ["f1_micro", "f1_macro", "f1_weighted", "jaccard_samples", "hamming_loss"]},
            },
        ]
    )

    joblib.dump(after_model, models_dir / "enhanced_tfidf.joblib")
    joblib.dump(labels, models_dir / "enhanced_labels.joblib")
    joblib.dump(after_label_thresholds, models_dir / "enhanced_thresholds.joblib")
    joblib.dump(
        {
            "before": before_metrics,
            "after_global": after_global_metrics,
            "after_per_label": after_label_metrics,
            "summary": metrics_summary,
            "labels": labels,
            "before_threshold": before_threshold,
            "after_global_threshold": after_global_threshold,
            "after_label_thresholds": after_label_thresholds,
        },
        models_dir / "enhanced_metrics.joblib",
    )

    metrics_summary.to_csv(reports_dir / "enhanced_before_after_metrics.csv", index=False)
    summarize_labels(current).to_csv(reports_dir / "current_grouped_label_counts.csv", index=False)
    summarize_labels(enriched).to_csv(reports_dir / "mal_manga_label_counts.csv", index=False)

    metadata = {
        "include_anime": args.include_anime,
        "current_rows": len(current),
        "mal_rows": len(enriched),
        "current_train_rows": len(current_train),
        "current_valid_rows": len(current_valid),
        "current_test_rows": len(current_test),
        "labels": labels,
    }
    (reports_dir / "enhanced_experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(metrics_summary.to_string(index=False))
    print("\nSaved enhanced model artifacts in:", models_dir)
    print("Saved reports in:", reports_dir)


if __name__ == "__main__":
    main()
