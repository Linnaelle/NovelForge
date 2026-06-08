"""Download NovelForge model artifacts from Hugging Face when needed."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_MODEL_REPO_ID = "Linnaelle/NovelForge"


def get_model_repo_id(repo_id: str | None = None) -> str:
    """Return the configured Hugging Face model repository."""
    return repo_id or os.getenv("NOVELFORGE_MODEL_REPO") or DEFAULT_MODEL_REPO_ID


def get_hf_token(token: str | None = None) -> str | None:
    """Return a Hugging Face token from an explicit value or environment."""
    return (
        token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )


def ensure_model_file(
    filename: str,
    target_path: Path,
    *,
    repo_id: str | None = None,
    token: str | None = None,
) -> Path | None:
    """Ensure a model file exists locally, downloading it from Hugging Face if missing."""
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded_path = hf_hub_download(
            repo_id=get_model_repo_id(repo_id),
            filename=filename,
            repo_type="model",
            token=get_hf_token(token),
            local_dir=target_path.parent,
        )
    except Exception:
        return None

    downloaded = Path(downloaded_path)
    return downloaded if downloaded.exists() else None


def ensure_model_dir(
    dirname: str,
    target_dir: Path,
    *,
    repo_id: str | None = None,
    token: str | None = None,
) -> Path | None:
    """Ensure a model directory exists locally, downloading it from Hugging Face if missing."""
    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=get_model_repo_id(repo_id),
            repo_type="model",
            token=get_hf_token(token),
            local_dir=target_dir.parent,
            allow_patterns=[f"{dirname}/**"],
        )
    except Exception:
        return None

    return target_dir if target_dir.exists() and any(target_dir.iterdir()) else None
